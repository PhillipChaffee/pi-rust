//! The schema layer: serializable definition data, the generated
//! vocabulary, and the typed span starter, ported from upstream
//! `index.ts`.
//!
//! Upstream `defineTelemetrySchema` is an identity helper whose real work is
//! type-level: closed-set `values` narrow to literal unions, required flags
//! split struct fields from `Option`s, and `createTypedSpanStarter` binds
//! span names to their owning schema's attributes. Rust has no string
//! literal types, so the port's [`define_telemetry_schema!`](crate::define_telemetry_schema) generates a
//! module of zero-sized marker types that carry the same guarantees in the
//! trait system: a span's start attributes are the generated struct, closed
//! sets are generated enums, and unknown names or cross-schema attributes
//! are type errors instead of type errors in a different dressing.
//!
//! Two upstream contracts are reshaped, not copied:
//!
//! - Duplicate span names across schemas were a compile-time type error;
//!   [`create_typed_span_starter!`](crate::create_typed_span_starter) performs the same rejection as a
//!   const-evaluated uniqueness check over the schemas' [`TelemetrySchema::SPAN_NAMES`].
//! - Schema definitions are JSON-serializable runtime data upstream; the
//!   port keeps them as plain data ([`TelemetrySchemaDefinition`]) and leaves
//!   JSON emission to whichever crate needs it.

use std::collections::BTreeMap;

use crate::SpanAttributes;

/// One literal in a closed-set or example list.
#[derive(Clone, Debug, PartialEq)]
pub enum TelemetryLiteral {
    /// A string literal.
    Str(String),
    /// An integral literal.
    Int(i64),
    /// A floating-point literal.
    Float(f64),
    /// A boolean literal.
    Bool(bool),
}

/// The value kind of an attribute definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryAttributeType {
    /// A string attribute.
    Str,
    /// A numeric attribute.
    Number,
    /// A boolean attribute.
    Boolean,
    /// A list of strings.
    StrArray,
    /// A list of numbers.
    NumberArray,
    /// A list of booleans.
    BooleanArray,
}

/// How often a value repeats in practice, upstream's `cardinality`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryAttributeCardinality {
    /// The value repeats little across spans.
    Low,
    /// The value varies widely across spans.
    High,
}

/// The documentation metadata every attribute definition carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TelemetryAttributeMetadata {
    /// The attribute's documented meaning.
    pub description: String,
    /// Whether the attribute may carry sensitive content.
    pub sensitive: bool,
    /// Upstream's `cardinality`; absent upstream stays `None` here.
    pub cardinality: Option<TelemetryAttributeCardinality>,
}

/// One attribute of a span or event, as serializable data.
///
/// Upstream models this as a discriminated union over `type`; the port
/// flattens it to a struct where `values` holds the closed set (scalar
/// kinds) and `element_values` holds the array-element closed set. The
/// definition carries no runtime validation — like upstream, schemas are
/// type-level contracts.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryAttributeDefinition {
    /// Documentation and sensitivity metadata.
    pub metadata: TelemetryAttributeMetadata,
    /// The value kind this attribute accepts.
    pub kind: TelemetryAttributeType,
    /// The closed set of scalar values; empty when the attribute is open.
    pub values: Vec<TelemetryLiteral>,
    /// The closed set of array element values, for array kinds.
    pub element_values: Vec<TelemetryLiteral>,
    /// Example values for documentation tooling.
    pub examples: Vec<TelemetryLiteral>,
    /// Whether the attribute is required at its recording site.
    pub required: bool,
}

/// One event of a span.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryEventDefinition {
    /// The event's documented meaning.
    pub description: String,
    /// The event's attributes, by name.
    pub attributes: BTreeMap<String, TelemetryAttributeDefinition>,
}

/// Where a span may be parented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TelemetryParentDefinition {
    /// Any parent, including none.
    Any,
    /// A root or an externally supplied parent.
    RootOrExternal,
    /// One of the named spans.
    Spans(Vec<String>),
}

/// The status block of a span definition; `default: "ok"` is structural, so
/// only the failure documentation is carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TelemetryStatusDefinition {
    /// When the span is expected to settle with an error status.
    pub error_when: String,
}

/// One span of a schema.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetrySpanDefinition {
    /// The span's documented meaning.
    pub description: String,
    /// Where the span may attach.
    pub parents: TelemetryParentDefinition,
    /// Attributes required or allowed at start, by name.
    pub start_attributes: BTreeMap<String, TelemetryAttributeDefinition>,
    /// Attributes allowed at the end of the span, by name; all optional.
    pub end_attributes: BTreeMap<String, TelemetryAttributeDefinition>,
    /// Events the span may record, by name; empty when the span records none.
    pub events: BTreeMap<String, TelemetryEventDefinition>,
    /// The status block.
    pub status: TelemetryStatusDefinition,
}

/// A whole schema: the serializable definition data.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetrySchemaDefinition {
    /// The schema version, monotonically raised on vocabulary changes.
    pub version: u32,
    /// The schema's spans, by name.
    pub spans: BTreeMap<String, TelemetrySpanDefinition>,
}

/// A macro-generated schema marker: names its spans and exposes the
/// definition data.
pub trait TelemetrySchema {
    /// The span names owned by this schema, in declaration order.
    const SPAN_NAMES: &'static [&'static str];

    /// The serializable definition data; rebuilt per call.
    fn definition() -> TelemetrySchemaDefinition;
}

/// A macro-generated span marker: the compile-time name of one span.
pub trait SpanDefinition {
    /// The recorded span name.
    const NAME: &'static str;
    /// The typed start attributes.
    type Start: IntoSpanAttributes;
    /// The typed end attributes, all optional.
    type End: IntoSpanAttributes;
}

/// A macro-generated event marker bound to one span.
pub trait EventDefinition {
    /// The recorded event name.
    const NAME: &'static str;
    /// The typed event attributes.
    type Attributes: IntoSpanAttributes;
}

/// Marks which span an event belongs to; generated per span-event pair, so
/// events of other spans are type errors at the call site.
pub trait EventOf<Span> {}

/// Converts a generated attribute struct into raw span attributes.
pub trait IntoSpanAttributes {
    /// Converts the typed attributes, skipping absent (`None`) entries.
    fn into_span_attributes(self) -> SpanAttributes;
}

/// True when no span name appears twice, within or across the given
/// schemas' name lists; used by [`create_typed_span_starter!`](crate::create_typed_span_starter) as the
/// compile-time duplicate rejection.
#[must_use]
pub const fn span_names_unique(schemas: &[&[&'static str]]) -> bool {
    let mut i = 0;
    while i < schemas.len() {
        let mut a = 0;
        while a < schemas[i].len() {
            let mut j = i;
            while j < schemas.len() {
                let start = if j == i { a + 1 } else { 0 };
                let mut b = start;
                while b < schemas[j].len() {
                    if str_bytes_eq(schemas[i][a], schemas[j][b]) {
                        return false;
                    }
                    b += 1;
                }
                j += 1;
            }
            a += 1;
        }
        i += 1;
    }
    true
}

/// Compares two strings byte by byte; `PartialEq` is not callable in const
/// contexts yet, so the uniqueness check walks the bytes itself.
const fn str_bytes_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut index = 0;
    while index < a.len() {
        if a[index] != b[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// Emits the field type for one attribute entry: the closed-set alias when
/// `values` names an enum, otherwise the primitive for the kind, `Option`-
/// wrapped unless the entry says `required`. A `values` entry without an
/// alias degrades to the primitive type.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attr_field_type {
    (as $t:ident : string required $vg:tt) => {
        $t
    };
    (as $t:ident : string optional $vg:tt) => {
        Option<$t>
    };
    (: $kind:ident required $vg:tt) => {
        $crate::__telemetry_field_type!($kind)
    };
    (: $kind:ident optional $vg:tt) => {
        Option<$crate::__telemetry_field_type!($kind)>
    };
    (: $kind:ident required) => {
        $crate::__telemetry_field_type!($kind)
    };
    (: $kind:ident optional) => {
        Option<$crate::__telemetry_field_type!($kind)>
    };
}

/// Maps an attribute kind token to the data-model kind.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_kind {
    (string) => {
        $crate::schema::TelemetryAttributeType::Str
    };
    (number) => {
        $crate::schema::TelemetryAttributeType::Number
    };
    (boolean) => {
        $crate::schema::TelemetryAttributeType::Boolean
    };
    (string_array) => {
        $crate::schema::TelemetryAttributeType::StrArray
    };
    (number_array) => {
        $crate::schema::TelemetryAttributeType::NumberArray
    };
    (boolean_array) => {
        $crate::schema::TelemetryAttributeType::BooleanArray
    };
}

/// Maps a plain attribute kind token to its field type.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_field_type {
    (string) => {
        String
    };
    (number) => {
        i64
    };
    (boolean) => {
        bool
    };
    (string_array) => {
        Vec<String>
    };
    (number_array) => {
        Vec<i64>
    };
    (boolean_array) => {
        Vec<bool>
    };
}

/// Emits the closed-set enum for one attribute with `values`.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attr_enum {
    (as $t:ident : string $mode:ident [ $($v:ident : $l:literal),* $(,)? ] description: $d:literal) => {
        #[doc = $d]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $t {
            $(
                #[doc = $l]
                $v,
            )*
        }

        impl $t {
            /// The recorded attribute value for this variant.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $( Self::$v => $l, )*
                }
            }
        }

        impl From<$t> for $crate::AttributeValue {
            fn from(value: $t) -> Self {
                $crate::AttributeValue::Str(value.as_str().to_owned())
            }
        }
    };
    ($($rest:tt)*) => {};
}

/// Emits the attribute iterator for one generated field; required entries
/// contribute one pair, optional entries contribute zero or one.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attr_iter {
    ($an:literal $f:ident required) => {
        ::std::iter::once(($an.to_owned(), Some($crate::AttributeValue::from($f))))
    };
    ($an:literal $f:ident optional) => {
        $f.map(|value| ($an.to_owned(), Some($crate::AttributeValue::from(value))))
    };
}

/// Emits the closed-set literal list, or an empty vec for plain attributes.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_values {
    ([ $($v:ident : $l:literal),* $(,)? ]) => {
        vec![ $( $crate::schema::TelemetryLiteral::Str($l.to_owned()) ),* ]
    };
    () => {
        Vec::new()
    };
}

/// Emits the `(name, definition)` pair for one attribute, for the data-map
/// iterator chain.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attr_data {
    ($an:literal : string required $vg:tt description: $d:literal) => {
        ::std::iter::once((
            $an.to_owned(),
            $crate::schema::TelemetryAttributeDefinition {
                metadata: $crate::schema::TelemetryAttributeMetadata {
                    description: $d.to_owned(),
                    sensitive: false,
                    cardinality: None,
                },
                kind: $crate::schema::TelemetryAttributeType::Str,
                values: $crate::__telemetry_values!($vg),
                element_values: Vec::new(),
                examples: Vec::new(),
                required: true,
            },
        ))
    };
    ($an:literal : string optional $vg:tt description: $d:literal) => {
        ::std::iter::once((
            $an.to_owned(),
            $crate::schema::TelemetryAttributeDefinition {
                metadata: $crate::schema::TelemetryAttributeMetadata {
                    description: $d.to_owned(),
                    sensitive: false,
                    cardinality: None,
                },
                kind: $crate::schema::TelemetryAttributeType::Str,
                values: $crate::__telemetry_values!($vg),
                element_values: Vec::new(),
                examples: Vec::new(),
                required: false,
            },
        ))
    };
    ($an:literal : $kind:ident required description: $d:literal) => {
        ::std::iter::once((
            $an.to_owned(),
            $crate::schema::TelemetryAttributeDefinition {
                metadata: $crate::schema::TelemetryAttributeMetadata {
                    description: $d.to_owned(),
                    sensitive: false,
                    cardinality: None,
                },
                kind: $crate::__telemetry_kind!($kind),
                values: Vec::new(),
                element_values: Vec::new(),
                examples: Vec::new(),
                required: true,
            },
        ))
    };
    ($an:literal : $kind:ident optional description: $d:literal) => {
        ::std::iter::once((
            $an.to_owned(),
            $crate::schema::TelemetryAttributeDefinition {
                metadata: $crate::schema::TelemetryAttributeMetadata {
                    description: $d.to_owned(),
                    sensitive: false,
                    cardinality: None,
                },
                kind: $crate::__telemetry_kind!($kind),
                values: Vec::new(),
                element_values: Vec::new(),
                examples: Vec::new(),
                required: false,
            },
        ))
    };
}

/// Emits the attribute struct, its conversion impl, and any closed-set
/// enums, from typed attribute entries.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attrs {
    (field $part:ident $subject:ident $vis:vis $name:ident { $( $an:literal $f:ident $(as $atype:ident)? : $kind:ident $mode:ident $(values $vg:tt)? description: $d:literal ),* $(,)? }) => {
        #[doc = concat!("Typed ", stringify!($part), " attributes for the ", stringify!($subject), ".")]
        $vis struct $name {
            $(
                #[doc = $d]
                pub $f: $crate::__telemetry_attr_field_type!($(as $atype)? : $kind $mode $( $vg )?),
            )*
        }

        impl $crate::schema::IntoSpanAttributes for $name {
            fn into_span_attributes(self) -> $crate::SpanAttributes {
                let Self { $( $f ),* } = self;
                $crate::SpanAttributes::from_iter(
                    ::std::iter::empty::<(String, Option<$crate::AttributeValue>)>()
                    $(
                        .chain( $crate::__telemetry_attr_iter!($an $f $mode) )
                    )*
                )
            }
        }

        $(
            $crate::__telemetry_attr_enum!($(as $atype)? : $kind $mode $( $vg )? description: $d);
        )*
    };
    (data { $( $an:literal $f:ident $(as $atype:ident)? : $kind:ident $mode:ident $(values $vg:tt)? description: $d:literal ),* $(,)? }) => {
        ::std::collections::BTreeMap::from_iter(
            ::std::iter::empty::<(String, $crate::schema::TelemetryAttributeDefinition)>()
            $(
                .chain( $crate::__telemetry_attr_data!($an : $kind $mode $( $vg )? description: $d) )
            )*
        )
    };
}

/// Emits one event module: the event marker, its typed attributes, and the
/// span binding.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_event {
    ($vis:vis $evn:ident $evd:literal { attributes: { $( $ean:literal $ef:ident $(as $etype:ident)? : $ekind:ident $emode:ident $(values $evg:tt)? description: $ed:literal ),* $(,)? } $(,)? }) => {
        #[doc = $evd]
        $vis mod $evn {
            #[doc = concat!("Event marker for the ", stringify!($evn), " event.")]
            pub struct Event;

            impl $crate::schema::EventDefinition for Event {
                const NAME: &'static str = stringify!($evn);
                type Attributes = Attributes;
            }

            impl $crate::schema::EventOf<super::Span> for Event {}

            $crate::__telemetry_attrs!(field event $evn pub Attributes {
                $( $ean $ef $(as $etype)? : $ekind $emode $(values $evg)? description: $ed ),*
            });
        }
    };
}

/// Emits the event modules for one span's captured events group.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_events {
    ({ $( $ven:ident => $ved:literal $attrs:tt ),* $(,)? }) => {
        $( $crate::__telemetry_event!(pub $ven $ved $attrs); )*
    };
}

/// Emits one span's wire name: the declared wire literal when present,
/// otherwise the span identifier.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_span_name {
    (as $wire:tt $($rest:tt)*) => { $wire };
    ($fallback:expr) => { $fallback };
}

/// Emits one span module: the span marker, typed start/end attribute
/// structs, and the event modules.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_span {
    ($vis:vis $span_name:ident $(as $span_wire:tt)? $span_desc:literal {
        parents: $p:tt,
        start_attributes: { $( $san:literal $sf:ident $(as $satype:ident)? : $skind:ident $smode:ident $(values $svg:tt)? description: $sd:literal ),* $(,)? },
        end_attributes: { $( $ean:literal $ef:ident $(as $eatype:ident)? : $ekind:ident $emode:ident $(values $evg:tt)? description: $ed:literal ),* $(,)? },
        $( events: $events:tt, )?
        status: { default ok, error_when $ew:literal } $(,)?
    }) => {
        #[doc = $span_desc]
        $vis mod $span_name {
            #[doc = concat!("Span marker for the ", stringify!($span_name), " span.")]
            pub struct Span;

            impl $crate::schema::SpanDefinition for Span {
                const NAME: &'static str =
                    $crate::__telemetry_span_name!($(as $span_wire)? stringify!($span_name));
                type Start = Start;
                type End = End;
            }

            $crate::__telemetry_attrs!(field start $span_name pub Start { $( $san $sf $(as $satype)? : $skind $smode $(values $svg)? description: $sd ),* }
            );

            $crate::__telemetry_attrs!(field end $span_name pub End { $( $ean $ef $(as $eatype)? : $ekind $emode $(values $evg)? description: $ed ),* }
            );

            $(
                $crate::__telemetry_events!($events);
            )*
        }
    };
}

/// Emits one span's `(name, definition)` data pair, for the schema data map.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_span_data {
    ($span_name:ident $(as $span_wire:tt)? $span_desc:literal {
        parents: $p:tt,
        start_attributes: { $( $san:literal $sf:ident $(as $satype:ident)? : $skind:ident $smode:ident $(values $svg:tt)? description: $sd:literal ),* $(,)? },
        end_attributes: { $( $ean:literal $ef:ident $(as $eatype:ident)? : $ekind:ident $emode:ident $(values $evg:tt)? description: $ed:literal ),* $(,)? },
        $( events: $events:tt, )?
        status: { default ok, error_when $ew:literal } $(,)?
    }) => {
        (
            $crate::__telemetry_span_name!($(as $span_wire)? stringify!($span_name)).to_owned(),
            $crate::schema::TelemetrySpanDefinition {
                description: $span_desc.to_owned(),
                parents: $crate::__telemetry_parents!($p),
                start_attributes: $crate::__telemetry_attrs!(data {
                    $( $san $sf $(as $satype)? : $skind $smode $(values $svg)? description: $sd ),*
                }),
                end_attributes: $crate::__telemetry_attrs!(data {
                    $( $ean $ef $(as $eatype)? : $ekind $emode $(values $evg)? description: $ed ),*
                }),
                events: $crate::__telemetry_events_data!($( $events )?),
                status: $crate::schema::TelemetryStatusDefinition {
                    error_when: $ew.to_owned(),
                },
            },
        )
    };
}

/// Emits the events map data, empty for spans that record no events.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_events_data {
    ({ $( $ven:ident => $ved:literal $attrs:tt ),* $(,)? }) => {
        ::std::collections::BTreeMap::from_iter(
            ::std::iter::empty::<(String, $crate::schema::TelemetryEventDefinition)>()
            $(
                .chain( ::std::iter::once((
                    stringify!($ven).to_owned(),
                    $crate::schema::TelemetryEventDefinition {
                        description: $ved.to_owned(),
                        attributes: $crate::__telemetry_attrs_data_group!($attrs),
                    },
                ))) )*
        )
    };
    () => {
        ::std::collections::BTreeMap::new()
    };
}

/// Emits one captured attributes group as data-map entries.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_attrs_data_group {
    ({ attributes: { $( $an:literal $f:ident $(as $atype:ident)? : $kind:ident $mode:ident $(values $vg:tt)? description: $d:literal ),* $(,)? } $(,)? }) => {
        $crate::__telemetry_attrs!(data {
            $( $an $f $(as $atype)? : $kind $mode $(values $vg)? description: $d ),*
        })
    };
}

/// Maps a parents token to the data model.
#[doc(hidden)]
#[macro_export]
macro_rules! __telemetry_parents {
    ([any]) => {
        $crate::schema::TelemetryParentDefinition::Any
    };
    ([root_or_external]) => {
        $crate::schema::TelemetryParentDefinition::RootOrExternal
    };
    ([spans $($p:literal),* $(,)?]) => {
        $crate::schema::TelemetryParentDefinition::Spans(vec![ $( $p.to_owned() ),* ])
    };
}

/// Defines one telemetry schema as a module of marker types plus the
/// serializable definition data, porting upstream's `defineTelemetrySchema`.
///
/// The generated vocabulary gives the compile-time guarantees upstream
/// carries in its type system: start attributes are the generated struct
/// (required entries are plain fields, optional ones are `Option`), closed
/// sets are generated enums, and events bind to their span through
/// [`crate::schema::EventOf`]. A `values` entry must name its enum with
/// `as`; without one the field degrades to the primitive type.
///
/// Grammar per attribute: `"name" field [as EnumType] : kind required|optional
/// [values [Variant: "value", ...]] description: "doc"`, where kind is one
/// of `string`, `number`, `boolean`, `string_array`, `number_array`,
/// `boolean_array`. Closed sets are string-only.
///
/// Grammar per span: `ident [as "wire-name"] => "description" { ... }`.
/// The Rust identifier names the generated module; the optional wire name
/// literal sets the span name the schema data and `SPAN_NAMES` carry, for
/// dotted wire names a Rust identifier cannot spell. Without it the span
/// name is the identifier.
///
/// # Examples
///
/// ```
/// use pi_telemetry::define_telemetry_schema;
/// use pi_telemetry::schema::TelemetrySchema;
///
/// define_telemetry_schema! {
///     /// The vocabulary of a test operation.
///     pub schema test_operation {
///         version: 1,
///         spans: {
///             /// The operation span.
///             operation => "Test operation" {
///                 parents: [any],
///                 start_attributes: {
///                     "kind" kind as Kind: string required values [Read: "read", Write: "write"] description: "Kind",
///                 },
///                 end_attributes: {},
///                 events: {
///                     result => "Result" {
///                         attributes: {
///                             "outcome" outcome as Outcome: string required values [Ok: "ok", Error: "error"] description: "Outcome",
///                         },
///                     },
///                 },
///                 status: { default ok, error_when "The operation fails" },
///             },
///         },
///     }
/// }
///
/// let schema = test_operation::Schema::definition();
/// assert_eq!(schema.version, 1);
/// let _span = test_operation::operation::Start {
///     kind: test_operation::operation::Kind::Read,
/// };
/// ```
///
/// Required event attributes cannot be omitted — as a compile error through
/// the typed starter:
///
/// Required event attributes cannot be omitted — as a compile error through
/// the typed starter:
///
/// ```compile_fail
/// use pi_telemetry::create_typed_span_starter;
/// use pi_telemetry::define_telemetry_schema;
/// use pi_telemetry::memory::InMemoryTelemetryContext;
///
/// define_telemetry_schema! {
///     /// The vocabulary of a test operation.
///     pub schema test_operation {
///         version: 1,
///         spans: {
///             /// The operation span.
///             operation => "Test operation" {
///                 parents: [any],
///                 start_attributes: {},
///                 end_attributes: {},
///                 events: {
///                     /// The result event.
///                     result => "Result" {
///                         attributes: {
///                             "outcome" outcome as Outcome: string required values [Ok: "ok", Error: "error"] description: "Outcome",
///                         },
///                     },
///                 },
///                 status: { default ok, error_when "The operation fails" },
///             },
///         },
///     }
/// }
///
/// async fn add_event_without_required_attributes() {
///     let context = InMemoryTelemetryContext::new();
///     let starter = create_typed_span_starter!(context, test_operation::Schema);
///     let _ = starter.start_span(
///         test_operation::operation::Span,
///         test_operation::operation::Start {},
///         |span, _child| async move {
///             // A required event attribute cannot be omitted.
///             span.add_event(test_operation::operation::result::Event, test_operation::operation::result::Attributes {});
///             Ok(())
///         },
///     );
/// }
/// ```
#[macro_export]
macro_rules! define_telemetry_schema {
    (
        $(#[$schema_meta:meta])*
        $schema_vis:vis schema $schema_name:ident {
            version: $version:literal,
            spans: {
                $(
                    $(#[$span_meta:meta])*
                    $span_name:ident $(as $span_wire:tt)? => $span_desc:literal {
                        parents: $p:tt,
                        start_attributes: { $( $san:literal $sf:ident $(as $satype:ident)? : $skind:ident $smode:ident $(values $svg:tt)? description: $sd:literal ),* $(,)? },
                        end_attributes: { $( $ean:literal $ef:ident $(as $eatype:ident)? : $ekind:ident $emode:ident $(values $evg:tt)? description: $ed:literal ),* $(,)? },
                        $( events: $events:tt, )?
                        status: { default ok, error_when $ew:literal } $(,)?
                    }
                ),*
                $(,)?
            } $(,)?
        }
    ) => {
        $(#[$schema_meta])*
        $schema_vis mod $schema_name {
            #[doc = concat!("Schema marker for the ", stringify!($schema_name), " vocabulary.")]
            $schema_vis struct Schema;

            impl $crate::schema::TelemetrySchema for Schema {
                const SPAN_NAMES: &'static [&'static str] =
                    &[ $( $crate::__telemetry_span_name!($(as $span_wire)? stringify!($span_name)) ),* ];

                fn definition() -> $crate::schema::TelemetrySchemaDefinition {
                    $crate::schema::TelemetrySchemaDefinition {
                        version: $version,
                        spans: ::std::collections::BTreeMap::from_iter(
                            ::std::iter::empty::<(String, $crate::schema::TelemetrySpanDefinition)>()
                            $(
                                .chain( ::std::iter::once( $crate::__telemetry_span_data!(
                                    $span_name $(as $span_wire)? $span_desc {
                                        parents: $p,
                                        start_attributes: { $( $san $sf $(as $satype)? : $skind $smode $(values $svg)? description: $sd ),* },
                                        end_attributes: { $( $ean $ef $(as $eatype)? : $ekind $emode $(values $evg)? description: $ed ),* },
                                        $( events: $events, )?
                                        status: { default ok, error_when $ew }
                                    }
                                ) ) ) )*
                        ),
                    }
                }
            }

            $(
                $(#[$span_meta])*
                $crate::__telemetry_span!(
                    $schema_vis $span_name $(as $span_wire)? $span_desc {
                        parents: $p,
                        start_attributes: { $( $san $sf $(as $satype)? : $skind $smode $(values $svg)? description: $sd ),* },
                        end_attributes: { $( $ean $ef $(as $eatype)? : $ekind $emode $(values $evg)? description: $ed ),* },
                        $( events: $events, )?
                        status: { default ok, error_when $ew }
                    }
                );
            )*
        }
    };
}

/// Creates a typed span starter over one or more schemas, rejecting
/// duplicate span names across the schemas at compile time — the port of
/// upstream's `createTypedSpanStarter` duplicate-schema check.
///
/// # Examples
///
/// ```
/// use pi_telemetry::{create_typed_span_starter, define_telemetry_schema};
/// use pi_telemetry::memory::InMemoryTelemetryContext;
///
/// define_telemetry_schema! {
///     /// The operation vocabulary.
///     pub schema test_operation {
///         version: 1,
///         spans: {
///             /// The operation span.
///             operation => "Operation" {
///                 parents: [root_or_external],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The operation fails" },
///             },
///         },
///     }
/// }
///
/// # fn main() {
/// let context = InMemoryTelemetryContext::new();
/// let starter = pi_telemetry::create_typed_span_starter!(context, test_operation::Schema);
/// # }
/// ```
///
/// Two schemas claiming the same span name are rejected at compile time:
///
/// ```compile_fail
/// use pi_telemetry::create_typed_span_starter;
/// use pi_telemetry::define_telemetry_schema;
/// use pi_telemetry::memory::InMemoryTelemetryContext;
///
/// define_telemetry_schema! {
///     /// The first schema.
///     pub schema first {
///         version: 1,
///         spans: {
///             /// The request span.
///             request => "Request" {
///                 parents: [any],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The request fails" },
///             },
///         },
///     }
/// }
///
/// define_telemetry_schema! {
///     /// The second schema, duplicating the request span name.
///     pub schema second {
///         version: 1,
///         spans: {
///             /// The request span again.
///             request => "Request" {
///                 parents: [any],
///                 start_attributes: {},
///                 end_attributes: {},
///                 status: { default ok, error_when "The request fails" },
///             },
///         },
///     }
/// }
///
/// let context = InMemoryTelemetryContext::new();
/// let _ = pi_telemetry::create_typed_span_starter!(context, first::Schema, second::Schema);
/// ```
#[macro_export]
macro_rules! create_typed_span_starter {
    ($context:expr $(, $schema:ty)+ $(,)?) => {{
        const _: () = assert!($crate::schema::span_names_unique(&[
            $( <$schema as $crate::schema::TelemetrySchema>::SPAN_NAMES ),+
        ]));
        $crate::typed::TypedSpanStarter::new($context)
    }};
}
