r"""Generate a Mermaid class diagram of ``src/pypft`` from the source itself.

Parses every module under a package root with ``ast`` -- nothing is imported, so
the diagram can be regenerated without a working environment -- and emits a
Mermaid ``classDiagram``: one box per class, one ``<<module>>`` box per module
that owns free functions or constants, grouped into ``namespace`` blocks by
package.

Four kinds of edge are derived, each from a distinct source construct:

- ``Base <|-- Derived`` from a class's own bases (in-package bases only; an
  external base such as ``Enum`` or ``UserWarning`` becomes a stereotype).
- ``Owner *-- Type`` from an annotated attribute whose type names an in-package
  class, with a ``"*"`` cardinality when that annotation is a container.
- ``Source ..> Type`` from a method's return annotation, and from a module-level
  function's parameter/return annotations.
- ``SomeEnum ..> Impl : selects`` from a module-level ``dict`` literal mapping
  enum members to classes -- the strategy-dispatch tables in ``pypft.dht`` and
  ``pypft.dft``.

Usage:
    uv run python scripts/make_class_diagram.py [--source src/pypft] \
        [--output <path>] [--direction TB] [--title <text>] [--no-namespaces] \
        [--no-private] [--no-module-functions] [--no-fence]

Writes to stdout unless ``--output`` is given, fenced as a ``mermaid`` block
(``--no-fence`` for a bare ``.mmd`` body). ``--no-private`` drops every
underscore-prefixed module, class, and member, leaving the public API surface
alone; the default keeps them, since the strategy implementations this package
is built around (``pypft.dht._naive`` and friends) all live in private modules.

Only the standard library is used, so this runs under a bare interpreter as well
as under ``uv run``.
"""

import argparse
import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ======================================================================================
# Extraction vocabulary
# ======================================================================================

#: Base-class names that mark a class as an enumeration. Matched by name, since
#: nothing is imported -- ``Axis(IntEnum)`` is recognized without resolving
#: ``enum.IntEnum`` itself.
_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "ReprEnum"})

#: Decorator names that make a method static (no instance is involved), rendered
#: with Mermaid's own ``$`` static classifier.
_STATIC_DECORATORS = frozenset({"staticmethod", "classmethod"})

#: Parameter names dropped from a rendered method signature, being an artifact of
#: the binding rather than part of the call.
_IMPLICIT_PARAMETERS = frozenset({"self", "cls"})

#: Annotation head names that make an attribute hold *many* of its element type,
#: rendered as a ``"*"`` cardinality on the composition edge.
_CONTAINER_NAMES = frozenset(
    {
        "dict",
        "frozenset",
        "Iterable",
        "list",
        "Mapping",
        "Sequence",
        "set",
        "tuple",
    }
)

#: Annotation names that never earn an edge of their own: they wrap or qualify
#: the type that does.
_TRANSPARENT_NAMES = frozenset({"ClassVar", "Final", "Optional", "Union"})

# ======================================================================================
# Rendering vocabulary
# ======================================================================================

#: Stereotype shown for each kind of box, in the priority order a class is
#: tested against (an enum that is also a dataclass reads as an enum).
_STEREOTYPE_ENUM = "enumeration"
_STEREOTYPE_WARNING = "warning"
_STEREOTYPE_EXCEPTION = "exception"
_STEREOTYPE_DATACLASS = "dataclass"
_STEREOTYPE_STATIC = "static"
_STEREOTYPE_MODULE = "module"

#: Mermaid visibility prefixes: PyPFT's single-underscore convention is the only
#: distinction the language has to draw here.
_PUBLIC = "+"
_PRIVATE = "-"

#: Mermaid's own static classifier, appended to a member line as its very last
#: character (the parser strips it from the end, before anything else).
_STATIC_CLASSIFIER = "$"

#: Relationship arrows, with the precedence used when two constructs would draw
#: an edge between the same pair of boxes -- the strongest statement wins.
_INHERITANCE = "<|--"
_COMPOSITION = "*--"
_DEPENDENCY = "..>"
_ARROW_PRECEDENCE = {_DEPENDENCY: 0, _COMPOSITION: 1, _INHERITANCE: 2}

#: How many function names a single dependency edge's label lists before it
#: is summarized with a count instead.
_MAX_LABEL_FUNCTIONS = 2

#: One indentation level of generated Mermaid.
_INDENT = "    "


# ======================================================================================
# Parsed entities
# ======================================================================================


@dataclass(frozen=True)
class Entity:
    """One box in the diagram: a class, or a module's free functions.

    :param name: The box's Mermaid identifier.
    :type name: str
    :param namespace: The package the box is grouped under.
    :type namespace: str
    :param module: The dotted module the box was parsed from.
    :type module: str
    :param lineno: The line the box was defined on, used only for ordering.
    :type lineno: int
    :param stereotype: The ``<<...>>`` annotation, if any.
    :type stereotype: str | None
    :param members: The rendered member lines, in source order.
    :type members: tuple[str, ...]
    :param bases: The class's own base names, unresolved.
    :type bases: tuple[str, ...]
    :param attributes: Each annotated attribute, as ``(name, annotation)``.
    :type attributes: tuple[tuple[str, str], ...]
    :param references: Every annotation the box's own functions name, as
        ``(function, annotation)`` pairs -- return annotations for a class, and
        parameter annotations as well for a ``<<module>>`` box.
    :type references: tuple[tuple[str, str], ...]

    """

    name: str
    namespace: str
    module: str
    lineno: int
    stereotype: str | None
    members: tuple[str, ...]
    bases: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    references: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Relation:
    """One edge between two boxes.

    :param source: The identifier the edge starts at.
    :type source: str
    :param arrow: The Mermaid arrow drawn between the two.
    :type arrow: str
    :param target: The identifier the edge ends at.
    :type target: str
    :param label: The edge's label, empty for an unlabelled edge.
    :type label: str
    :param many: Whether the target end carries a ``"*"`` cardinality.
    :type many: bool

    """

    source: str
    arrow: str
    target: str
    label: str
    many: bool


# ======================================================================================
# Annotation helpers
# ======================================================================================


def _name_of(node: ast.expr) -> str:
    """Return the trailing name of a dotted expression.

    :param node: The expression to name, e.g. a base class or a decorator.
    :type node: ast.expr
    :returns: ``"ndarray"`` for ``np.ndarray``, ``"Enum"`` for ``Enum``, and the
        unparsed source for anything else.
    :rtype: str

    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ast.unparse(node)


def _annotation_of(node: ast.expr | None) -> str:
    """Return an annotation's source text, or an empty string if it is absent.

    :param node: The annotation node, if the construct carried one.
    :type node: ast.expr | None
    :returns: The unparsed annotation.
    :rtype: str

    """
    return "" if node is None else ast.unparse(node)


def _referenced_names(annotation: str) -> set[str]:
    """Return every bare identifier an annotation mentions.

    Deliberately lexical rather than a resolved type: an edge is wanted for
    ``BaseSignal`` in ``tuple[BaseSignal, ...]``, in ``ClassVar[Domain]``, and in
    ``pypft.grid.PolarGrid`` alike, and every name that is not a box in the
    diagram is filtered out by the caller anyway.

    :param annotation: The annotation's source text.
    :type annotation: str
    :returns: The identifiers it mentions, minus the wrappers that only ever
        qualify another type.
    :rtype: set[str]

    """
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", annotation))
    return names - _TRANSPARENT_NAMES


def _is_many(annotation: str) -> bool:
    """Report whether an annotation holds many of its element type.

    :param annotation: The annotation's source text.
    :type annotation: str
    :returns: Whether the annotation mentions a container type.
    :rtype: bool

    """
    return bool(_referenced_names(annotation) & _CONTAINER_NAMES)


def _mermaid_type(annotation: str) -> str:
    """Render an annotation in Mermaid's own type syntax.

    Subscripts become Mermaid generics (``tuple[Axes, Axes]`` ->
    ``tuple~Axes, Axes~``) and dotted qualifiers are dropped (``np.ndarray`` ->
    ``ndarray``), since the package a type comes from is already carried by the
    namespace it is drawn in.

    :param annotation: The annotation's source text.
    :type annotation: str
    :returns: The Mermaid-ready type name.
    :rtype: str

    """
    text = annotation.replace("[", "~").replace("]", "~").replace("'", "")
    text = text.replace('"', "")
    text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", text)
    return " ".join(text.split())


# ======================================================================================
# Member rendering
# ======================================================================================


def _visibility(name: str) -> str:
    """Return the Mermaid visibility prefix for a member or class name.

    :param name: The member's own name.
    :type name: str
    :returns: ``"-"`` for an underscore-prefixed name, ``"+"`` otherwise.
    :rtype: str

    """
    return _PRIVATE if name.startswith("_") else _PUBLIC


def _is_private(name: str) -> bool:
    """Report whether a name is private by PyPFT's underscore convention.

    :param name: The module, class, or member name.
    :type name: str
    :returns: Whether the name starts with an underscore.
    :rtype: bool

    """
    return name.startswith("_")


def _field_line(name: str, annotation: str, *, static: bool) -> str:
    """Render one attribute as a Mermaid field line.

    :param name: The attribute's name.
    :type name: str
    :param annotation: The attribute's annotation, possibly empty.
    :type annotation: str
    :param static: Whether the attribute belongs to the class, not an instance.
    :type static: bool
    :returns: The rendered line, e.g. ``"+ndarray values"``.
    :rtype: str

    """
    rendered = _mermaid_type(annotation)
    prefix = f"{_visibility(name)}{rendered} " if rendered else _visibility(name)
    return f"{prefix}{name}{_STATIC_CLASSIFIER if static else ''}"


def _method_line(node: ast.FunctionDef | ast.AsyncFunctionDef, *, static: bool) -> str:
    """Render one method or function as a Mermaid method line.

    Parameter *names* are listed without their own types: every signature in this
    package is annotated, and repeating those types inside the box makes it far
    wider than the information is worth.

    :param node: The function definition to render.
    :type node: ast.FunctionDef | ast.AsyncFunctionDef
    :param static: Whether the function is bound to the class, not an instance.
    :type static: bool
    :returns: The rendered line, e.g. ``"+forward(f, n, R, axis) ndarray$"``.
    :rtype: str

    """
    arguments = node.args
    names = [
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        if argument.arg not in _IMPLICIT_PARAMETERS
    ]
    if arguments.vararg is not None:
        names.append(f"*{arguments.vararg.arg}")
    if arguments.kwarg is not None:
        names.append(f"**{arguments.kwarg.arg}")
    returns = _mermaid_type(_annotation_of(node.returns))
    suffix = "" if returns in ("", "None") else f" {returns}"
    classifier = _STATIC_CLASSIFIER if static else ""
    signature = f"{node.name}({', '.join(names)})"
    return f"{_visibility(node.name)}{signature}{suffix}{classifier}"


# ======================================================================================
# Class extraction
# ======================================================================================


def _stereotype(
    node: ast.ClassDef, bases: tuple[str, ...], *, stateless: bool
) -> str | None:
    """Classify a class into the one stereotype Mermaid can show for it.

    :param node: The class definition.
    :type node: ast.ClassDef
    :param bases: The class's own base names.
    :type bases: tuple[str, ...]
    :param stateless: Whether every one of the class's methods is a
        ``staticmethod`` or a ``classmethod``, so it is never instantiated.
    :type stateless: bool
    :returns: The stereotype name, without its angle brackets, or ``None``.
    :rtype: str | None

    """
    decorators = {_name_of(decorator) for decorator in node.decorator_list}
    if set(bases) & _ENUM_BASES:
        return _STEREOTYPE_ENUM
    if any(base.endswith("Warning") for base in bases):
        return _STEREOTYPE_WARNING
    if any(base.endswith(("Error", "Exception")) for base in bases):
        return _STEREOTYPE_EXCEPTION
    if "dataclass" in decorators:
        return _STEREOTYPE_DATACLASS
    if stateless:
        return _STEREOTYPE_STATIC
    return None


def _class_entity(
    node: ast.ClassDef, module: str, namespace: str, *, private: bool
) -> Entity:
    """Extract one class into an ``Entity``.

    :param node: The class definition.
    :type node: ast.ClassDef
    :param module: The dotted module the class is defined in.
    :type module: str
    :param namespace: The package the class is grouped under.
    :type namespace: str
    :param private: Whether underscore-prefixed members are kept.
    :type private: bool
    :returns: The extracted box.
    :rtype: Entity

    """
    bases = tuple(_name_of(base) for base in node.bases)
    is_enum = bool(set(bases) & _ENUM_BASES)
    members: list[str] = []
    attributes: list[tuple[str, str]] = []
    references: list[tuple[str, str]] = []
    methods = 0
    stateless = 0
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            annotation = _annotation_of(item.annotation)
            attributes.append((name, annotation))
            if private or not _is_private(name):
                static = annotation.startswith("ClassVar")
                members.append(_field_line(name, annotation, static=static))
        elif isinstance(item, ast.Assign) and is_enum:
            for target in item.targets:
                if isinstance(target, ast.Name) and (
                    private or not _is_private(target.id)
                ):
                    members.append(target.id)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {_name_of(decorator) for decorator in item.decorator_list}
            methods += 1
            static = bool(decorators & _STATIC_DECORATORS)
            stateless += int(static)
            returns = _annotation_of(item.returns)
            references.append((item.name, returns))
            if not private and _is_private(item.name):
                continue
            if "property" in decorators:
                members.append(_field_line(item.name, returns, static=False))
            else:
                members.append(_method_line(item, static=static))
    all_stateless = methods > 0 and stateless == methods
    return Entity(
        name=node.name,
        namespace=namespace,
        module=module,
        lineno=node.lineno,
        stereotype=_stereotype(node, bases, stateless=all_stateless),
        members=tuple(members),
        bases=bases,
        attributes=tuple(attributes),
        references=tuple(references),
    )


# ======================================================================================
# Module extraction
# ======================================================================================


def _module_entity(
    tree: ast.Module, name: str, module: str, namespace: str, *, private: bool
) -> Entity | None:
    """Extract a module's free functions and constants into a ``<<module>>`` box.

    A module whose body is nothing but class definitions gets no box of its own:
    its classes already carry everything it has to say.

    :param tree: The parsed module.
    :type tree: ast.Module
    :param name: The box's Mermaid identifier.
    :type name: str
    :param module: The dotted module name.
    :type module: str
    :param namespace: The package the box is grouped under.
    :type namespace: str
    :param private: Whether underscore-prefixed members are kept.
    :type private: bool
    :returns: The extracted box, or ``None`` if the module defines no free
        function or constant worth showing.
    :rtype: Entity | None

    """
    members: list[str] = []
    references: list[tuple[str, str]] = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = [_annotation_of(item.returns)]
            annotations.extend(
                _annotation_of(argument.annotation)
                for argument in (*item.args.args, *item.args.kwonlyargs)
            )
            references.extend((item.name, annotation) for annotation in annotations)
            if private or not _is_private(item.name):
                members.append(_method_line(item, static=True))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            constant = item.target.id
            if constant.isupper() and (private or not _is_private(constant)):
                annotation = _annotation_of(item.annotation)
                members.append(_field_line(constant, annotation, static=True))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                    and (private or not _is_private(target.id))
                ):
                    members.append(_field_line(target.id, "", static=True))
    if not members:
        return None
    return Entity(
        name=name,
        namespace=namespace,
        module=module,
        lineno=0,
        stereotype=_STEREOTYPE_MODULE,
        members=tuple(members),
        bases=(),
        attributes=(),
        references=tuple(references),
    )


def _dispatch_relations(tree: ast.Module, known: set[str]) -> list[Relation]:
    """Derive the strategy-dispatch edges from a module's own lookup tables.

    A module-level ``dict`` literal keyed by enum members and valued by classes
    is exactly how ``pypft.dht``/``pypft.dft`` map a ``*Implementation`` member
    onto the class that implements it; each such pair becomes one edge.

    :param tree: The parsed module.
    :type tree: ast.Module
    :param known: Every identifier that has a box in the diagram.
    :type known: set[str]
    :returns: One ``selects`` edge per enum-member-to-class entry.
    :rtype: list[Relation]

    """
    relations: list[Relation] = []
    for item in ast.walk(tree):
        if not isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        if not isinstance(item.value, ast.Dict):
            continue
        for key, value in zip(item.value.keys, item.value.values):
            if not isinstance(key, ast.Attribute) or not isinstance(value, ast.Name):
                continue
            source = _name_of(key.value)
            if source in known and value.id in known:
                relations.append(
                    Relation(
                        source=source,
                        arrow=_DEPENDENCY,
                        target=value.id,
                        label="selects",
                        many=False,
                    )
                )
    return relations


# ======================================================================================
# Traversal
# ======================================================================================


def _iter_sources(source: Path, *, private: bool) -> Iterator[tuple[Path, str, str]]:
    """Walk a package root, yielding each module's path, dotted name, and package.

    :param source: The package root, e.g. ``src/pypft``.
    :type source: Path
    :param private: Whether underscore-prefixed modules are kept.
    :type private: bool
    :returns: One ``(path, module, namespace)`` triple per module, in path order.
    :rtype: Iterator[tuple[Path, str, str]]

    """
    root = source.parent
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        namespace = ".".join(path.parent.relative_to(root).parts)
        if not private and any(
            _is_private(part) for part in module.split(".")
        ):  # a private module, e.g. pypft.dht._naive
            continue
        yield path, module, namespace


def _module_box_names(modules: list[tuple[Path, str, str]]) -> dict[str, str]:
    """Choose a unique box identifier for each module's ``<<module>>`` box.

    The module's own basename is used wherever it is unambiguous (``transform``,
    ``grid``), falling back to the full underscore-joined path for a basename two
    packages share (``pypft.dht._base`` and ``pypft.dft._base``).

    :param modules: Every ``(path, module, namespace)`` triple being drawn.
    :type modules: list[tuple[Path, str, str]]
    :returns: The box identifier for each dotted module name.
    :rtype: dict[str, str]

    """
    basenames: dict[str, int] = {}
    for _, module, _ in modules:
        basename = module.rsplit(".", 1)[-1]
        basenames[basename] = basenames.get(basename, 0) + 1
    names = {}
    for _, module, _ in modules:
        basename = module.rsplit(".", 1)[-1]
        unique = basenames[basename] == 1
        names[module] = basename if unique else module.replace(".", "_")
    return names


def parse_package(
    source: Path, *, private: bool, module_functions: bool
) -> tuple[list[Entity], list[Relation]]:
    """Parse every module under a package root into boxes and edges.

    :param source: The package root, e.g. ``src/pypft``.
    :type source: Path
    :param private: Whether underscore-prefixed modules, classes, and members
        are kept.
    :type private: bool
    :param module_functions: Whether a ``<<module>>`` box is drawn for each
        module's own free functions and constants.
    :type module_functions: bool
    :returns: Every box, in source order, and the strategy-dispatch edges (the
        rest are derived once every box is known).
    :rtype: tuple[list[Entity], list[Relation]]
    :raises FileNotFoundError: If ``source`` is not an existing directory.

    """
    if not source.is_dir():
        raise FileNotFoundError(f"no package directory at {source}")
    modules = list(_iter_sources(source, private=private))
    box_names = _module_box_names(modules)
    entities: list[Entity] = []
    trees: list[ast.Module] = []
    for path, module, namespace in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees.append(tree)
        for item in tree.body:
            if not isinstance(item, ast.ClassDef):
                continue
            if private or not _is_private(item.name):
                entity = _class_entity(item, module, namespace, private=private)
                entities.append(entity)
        if module_functions:
            entity = _module_entity(
                tree, box_names[module], module, namespace, private=private
            )
            if entity is not None:
                entities.append(entity)
    known = {entity.name for entity in entities}
    dispatch = [
        relation for tree in trees for relation in _dispatch_relations(tree, known)
    ]
    return entities, dispatch


# ======================================================================================
# Relationships
# ======================================================================================


def _label(functions: list[str]) -> str:
    """Summarize the functions behind one dependency edge into a label.

    :param functions: Every function or method name that draws the same edge.
    :type functions: list[str]
    :returns: The names themselves while there are few, else how many there are.
    :rtype: str

    """
    if len(functions) <= _MAX_LABEL_FUNCTIONS:
        return "/".join(functions)
    return f"{len(functions)} functions"


def relationships(entities: list[Entity], dispatch: list[Relation]) -> list[Relation]:
    """Derive every edge between the diagram's boxes.

    Each pair of boxes keeps only its strongest edge: an inheritance already
    says more than the composition or dependency the same pair would also draw.

    :param entities: Every box in the diagram.
    :type entities: list[Entity]
    :param dispatch: The strategy-dispatch edges parsed alongside the boxes.
    :type dispatch: list[Relation]
    :returns: The edges to draw, ordered by source then target.
    :rtype: list[Relation]

    """
    known = {entity.name for entity in entities}
    best: dict[tuple[str, str], Relation] = {}

    def keep(relation: Relation) -> None:
        """Record an edge unless a stronger one already joins the same pair."""
        if relation.source == relation.target:
            return
        key = (relation.source, relation.target)
        current = best.get(key)
        if current is None or (
            _ARROW_PRECEDENCE[relation.arrow] > _ARROW_PRECEDENCE[current.arrow]
        ):
            best[key] = relation

    for entity in entities:
        for base in entity.bases:
            if base in known:
                keep(
                    Relation(
                        source=base,
                        arrow=_INHERITANCE,
                        target=entity.name,
                        label="",
                        many=False,
                    )
                )
        for name, annotation in entity.attributes:
            # A ClassVar tags the class itself rather than being held by an
            # instance of it, so it never earns a composition edge.
            tag = annotation.startswith("ClassVar")
            for referenced in sorted(_referenced_names(annotation) & known):
                keep(
                    Relation(
                        source=entity.name,
                        arrow=_DEPENDENCY if tag else _COMPOSITION,
                        target=referenced,
                        label=name,
                        many=not tag and _is_many(annotation),
                    )
                )
        named_by: dict[str, list[str]] = {}
        for function, annotation in entity.references:
            for referenced in sorted(_referenced_names(annotation) & known):
                if function not in named_by.setdefault(referenced, []):
                    named_by[referenced].append(function)
        for referenced, functions in named_by.items():
            keep(
                Relation(
                    source=entity.name,
                    arrow=_DEPENDENCY,
                    target=referenced,
                    label=_label(functions),
                    many=False,
                )
            )
    for relation in dispatch:
        keep(relation)
    return [best[key] for key in sorted(best)]


# ======================================================================================
# Mermaid rendering
# ======================================================================================


def _render_entity(entity: Entity, indent: str) -> list[str]:
    """Render one box, including its stereotype and members.

    :param entity: The box to render.
    :type entity: Entity
    :param indent: The leading whitespace every line of the box carries.
    :type indent: str
    :returns: The box's own lines.
    :rtype: list[str]

    """
    body = list(entity.members)
    if entity.stereotype is not None:
        body.insert(0, f"<<{entity.stereotype}>>")
    if not body:
        return [f"{indent}class {entity.name}"]
    lines = [f"{indent}class {entity.name} {{"]
    lines.extend(f"{indent}{_INDENT}{line}" for line in body)
    lines.append(f"{indent}}}")
    return lines


def _render_relation(relation: Relation) -> str:
    """Render one edge.

    :param relation: The edge to render.
    :type relation: Relation
    :returns: The edge's Mermaid line.
    :rtype: str

    """
    arrow = relation.arrow
    if relation.many:
        arrow = f'"1" {arrow} "*"'
    line = f"{_INDENT}{relation.source} {arrow} {relation.target}"
    return f"{line} : {relation.label}" if relation.label else line


def render(
    entities: list[Entity],
    relations: list[Relation],
    *,
    direction: str,
    namespaces: bool,
    title: str | None,
    source: Path,
) -> str:
    """Render the whole diagram.

    :param entities: Every box to draw.
    :type entities: list[Entity]
    :param relations: Every edge to draw.
    :type relations: list[Relation]
    :param direction: The Mermaid layout direction.
    :type direction: str
    :param namespaces: Whether boxes are grouped into ``namespace`` blocks by
        package.
    :type namespaces: bool
    :param title: The diagram's front-matter title, if any.
    :type title: str | None
    :param source: The package root the diagram was generated from, named in the
        generated header.
    :type source: Path
    :returns: The Mermaid source, without a surrounding code fence.
    :rtype: str

    """
    lines: list[str] = []
    if title is not None:
        lines.extend(["---", f"title: {title}", "---"])
    lines.append("classDiagram")
    lines.append(f"{_INDENT}direction {direction}")
    lines.append(
        f"{_INDENT}%% Generated from {source.as_posix()} by "
        "scripts/make_class_diagram.py -- do not edit by hand."
    )
    lines.append("")
    ordered = sorted(entities, key=lambda entity: (entity.namespace, entity.module))
    if namespaces:
        for namespace in sorted({entity.namespace for entity in ordered}):
            lines.append(f"{_INDENT}namespace {namespace} {{")
            for entity in ordered:
                if entity.namespace == namespace:
                    lines.extend(_render_entity(entity, indent=_INDENT * 2))
            lines.append(f"{_INDENT}}}")
            lines.append("")
    else:
        for entity in ordered:
            lines.extend(_render_entity(entity, indent=_INDENT))
        lines.append("")
    lines.extend(_render_relation(relation) for relation in relations)
    return "\n".join(lines) + "\n"


# ======================================================================================
# Entry point
# ======================================================================================


def _parse_arguments() -> argparse.Namespace:
    """Parse this script's own command line.

    :returns: The parsed arguments.
    :rtype: argparse.Namespace

    """
    parser = argparse.ArgumentParser(
        description="Generate a Mermaid class diagram of a package, by parsing it."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/pypft"),
        help="Package root to parse (default: src/pypft).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the diagram (default: stdout).",
    )
    parser.add_argument(
        "--direction",
        choices=("TB", "BT", "LR", "RL"),
        default="TB",
        help="Mermaid layout direction (default: TB).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Front-matter title for the diagram (default: none).",
    )
    parser.add_argument(
        "--namespaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group boxes into namespace blocks by package (default: on).",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include underscore-prefixed modules, classes, and members "
        "(default: on).",
    )
    parser.add_argument(
        "--module-functions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a <<module>> box for each module's free functions and "
        "constants (default: on).",
    )
    parser.add_argument(
        "--fence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the output in a ```mermaid code fence (default: on).",
    )
    return parser.parse_args()


def main() -> None:
    """Parse the package, render the diagram, and write it out."""
    args = _parse_arguments()
    entities, dispatch = parse_package(
        args.source, private=args.private, module_functions=args.module_functions
    )
    relations = relationships(entities, dispatch)
    diagram = render(
        entities,
        relations,
        direction=args.direction,
        namespaces=args.namespaces,
        title=args.title,
        source=args.source,
    )
    if args.fence:
        diagram = f"```mermaid\n{diagram}```\n"
    if args.output is None:
        sys.stdout.write(diagram)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(diagram, encoding="utf-8")
    print(
        f"Wrote {args.output} ({len(entities)} boxes, {len(relations)} relationships)"
    )


if __name__ == "__main__":
    main()
