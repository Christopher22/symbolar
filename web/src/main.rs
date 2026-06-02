use dioxus::prelude::*;
use dioxus_components::{
    Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Empty,
    EmptyDescription, EmptyHeader, EmptyTitle,
};
use std::num::NonZeroUsize;
use symbolar::{
    Dynamic, Expression, Storage,
    architectures::{
        BinarySpatterCode, HolographicReducedRepresentation, MultiplyAddPermute,
        VectorDerivedTransformationBinding,
    },
};

const FAVICON: Asset = asset!("/assets/favicon.ico");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

const MIN_DIMENSIONS: usize = 5000;
const MAX_DIMENSIONS: usize = 50_000;
const DIMENSION_STEP: usize = 128;
const DEFAULT_DIMENSIONS: usize = 10000;

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Architecture {
    Bsc,
    Map,
    Hrr,
    Vtb,
}

impl Architecture {
    fn as_value(self) -> &'static str {
        match self {
            Self::Bsc => "bsc",
            Self::Hrr => "hrr",
            Self::Map => "map",
            Self::Vtb => "vtb",
        }
    }
}

impl std::str::FromStr for Architecture {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bsc" => Ok(Self::Bsc),
            "hrr" => Ok(Self::Hrr),
            "map" => Ok(Self::Map),
            "vtb" => Ok(Self::Vtb),
            _ => return Err(()),
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_value())
    }
}

enum AppStorage {
    Bsc(Storage<Dynamic, BinarySpatterCode<u8>>),
    Hrr(Storage<Dynamic, HolographicReducedRepresentation<f64>>),
    Map(Storage<Dynamic, MultiplyAddPermute<usize, isize>>),
    Vtb(Storage<Dynamic, VectorDerivedTransformationBinding<f64>>),
}

impl AppStorage {
    fn new(architecture: Architecture, dimensions: usize) -> Self {
        let size = Dynamic::from(NonZeroUsize::new(dimensions).expect("dimensions must be > 0"));

        match architecture {
            Architecture::Bsc => {
                Self::Bsc(Storage::new(BinarySpatterCode::default(), size).expect("valid BSC size"))
            }
            Architecture::Hrr => Self::Hrr(
                Storage::new(HolographicReducedRepresentation::default(), size)
                    .expect("valid HRR size"),
            ),
            Architecture::Map => Self::Map(
                Storage::new(MultiplyAddPermute::default(), size).expect("valid MAP size"),
            ),
            Architecture::Vtb => Self::Vtb(
                Storage::new(VectorDerivedTransformationBinding::default(), size)
                    .expect("valid VTB size"),
            ),
        }
    }

    fn has(&self, name: &str) -> bool {
        match self {
            Self::Bsc(storage) => storage.get(&name).is_some(),
            Self::Hrr(storage) => storage.get(&name).is_some(),
            Self::Map(storage) => storage.get(&name).is_some(),
            Self::Vtb(storage) => storage.get(&name).is_some(),
        }
    }

    fn push(&mut self, name: String) {
        match self {
            Self::Bsc(storage) => {
                storage.push(name);
            }
            Self::Hrr(storage) => {
                storage.push(name);
            }
            Self::Map(storage) => {
                storage.push(name);
            }
            Self::Vtb(storage) => {
                storage.push(name);
            }
        }
    }

    fn similarities(&self, names: &[String], query_text: &str) -> Result<Vec<Option<f64>>, String> {
        let query_text = query_text.trim();
        if query_text.is_empty() {
            return Ok(vec![None; names.len()]);
        }

        let expression = query_text
            .parse::<Expression>()
            .map_err(|err| format!("Invalid query: {err}"))?;

        match self {
            Self::Bsc(storage) => {
                let query_vector = storage
                    .execute(&expression)
                    .map_err(|err| format!("Unknown value in query: {err}"))?
                    .into_owned();

                Ok(names
                    .iter()
                    .map(|name| {
                        storage
                            .get(&name.as_str())
                            .map(|item| item.similarity(&query_vector))
                    })
                    .collect())
            }
            Self::Hrr(storage) => {
                let query_vector = storage
                    .execute(&expression)
                    .map_err(|err| format!("Unknown value in query: {err}"))?
                    .into_owned();

                Ok(names
                    .iter()
                    .map(|name| {
                        storage
                            .get(&name.as_str())
                            .map(|item| item.similarity(&query_vector))
                    })
                    .collect())
            }
            Self::Map(storage) => {
                let query_vector = storage
                    .execute(&expression)
                    .map_err(|err| format!("Unknown value in query: {err}"))?
                    .into_owned();

                Ok(names
                    .iter()
                    .map(|name| {
                        storage
                            .get(&name.as_str())
                            .map(|item| item.similarity(&query_vector))
                    })
                    .collect())
            }
            Self::Vtb(storage) => {
                let query_vector = storage
                    .execute(&expression)
                    .map_err(|err| format!("Unknown value in query: {err}"))?
                    .into_owned();
                Ok(names
                    .iter()
                    .map(|name| {
                        storage
                            .get(&name.as_str())
                            .map(|item| item.similarity(&query_vector))
                    })
                    .collect())
            }
        }
    }
}

const COOLWARM_STOPS: [(u8, u8, u8); 11] = [
    (59, 76, 192),
    (84, 116, 224),
    (128, 165, 251),
    (171, 198, 254),
    (211, 223, 253),
    (221, 221, 221),
    (236, 211, 197),
    (245, 184, 156),
    (239, 138, 98),
    (214, 82, 66),
    (180, 4, 38),
];

fn lerp_channel(a: u8, b: u8, t: f64) -> u8 {
    let value = (a as f64) + ((b as f64) - (a as f64)) * t;
    value.round().clamp(0.0, 255.0) as u8
}

fn coolwarm_rgb(score: f64) -> (u8, u8, u8) {
    let clamped = score.clamp(-1.0, 1.0);
    let t = (clamped + 1.0) * 0.5;
    let n = COOLWARM_STOPS.len() - 1;
    let scaled = t * n as f64;
    let i = scaled.floor() as usize;

    if i >= n {
        return COOLWARM_STOPS[n];
    }

    let local_t = scaled - i as f64;
    let (r0, g0, b0) = COOLWARM_STOPS[i];
    let (r1, g1, b1) = COOLWARM_STOPS[i + 1];

    (
        lerp_channel(r0, r1, local_t),
        lerp_channel(g0, g1, local_t),
        lerp_channel(b0, b1, local_t),
    )
}

fn darken((r, g, b): (u8, u8, u8), factor: f64) -> (u8, u8, u8) {
    let scale = factor.clamp(0.0, 1.0);
    (
        ((r as f64) * scale).round() as u8,
        ((g as f64) * scale).round() as u8,
        ((b as f64) * scale).round() as u8,
    )
}

fn relative_luminance((r, g, b): (u8, u8, u8)) -> f64 {
    fn channel(v: u8) -> f64 {
        let x = (v as f64) / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

fn similarity_badge(similarity: Option<f64>) -> (String, String) {
    let label = similarity
        .map(|score| format!("sim = {score:.3}"))
        .unwrap_or_else(|| "sim = -".to_string());

    let style = match similarity {
        None => "border-color: rgb(203 213 225); background-color: rgb(241 245 249); color: rgb(51 65 85);".to_string(),
        Some(score) => {
            let color = coolwarm_rgb(score);
            let border = darken(color, 0.78);
            let text = if relative_luminance(color) > 0.45 {
                "rgb(15 23 42)"
            } else {
                "rgb(248 250 252)"
            };

            format!(
                "border-color: rgb({} {} {}); background-color: rgb({} {} {}); color: {};",
                border.0, border.1, border.2, color.0, color.1, color.2, text
            )
        }
    };

    (label, style)
}

fn similarity_tile_style(similarity: Option<f64>) -> String {
    match similarity {
        None => "border-color: rgb(226 232 240); background-color: rgb(255 255 255);".to_string(),
        Some(score) => {
            let color = coolwarm_rgb(score);
            let border = darken(color, 0.75);
            format!(
                "border-color: rgb({} {} {}); background-color: color-mix(in oklab, rgb({} {} {}) 18%, white);",
                border.0, border.1, border.2, color.0, color.1, color.2
            )
        }
    }
}

#[component]
fn ControlSidebar(
    architecture: Signal<Architecture>,
    dimensions: Signal<usize>,
    new_item: Signal<String>,
    item_count: usize,
    on_architecture_change: EventHandler<String>,
    on_dimensions_change: EventHandler<String>,
    on_add_item: EventHandler<()>,
) -> Element {
    rsx! {
        Card { class: "h-full p-3",
            CardHeader {
                CardTitle { "Symbolar: VSA Playground" }
                CardDescription {
                    "Add elements on the left, inspect them in a 2D grid, and enter a query at the bottom of the page."
                }
            }
            CardContent { class: "h-full",
                div { class: "space-y-5",
                    div {
                        label { class: "mb-1 block text-sm font-medium text-slate-700",
                            "Architecture"
                        }
                        select {
                            class: "w-full rounded-xl border border-slate-300 bg-white px-3 py-2 outline-none ring-offset-2 focus:border-slate-500 focus:ring-2 focus:ring-slate-400",
                            value: "{architecture().as_value()}",
                            onchange: move |event| on_architecture_change.call(event.value()),
                            option { value: "bsc", "Binary Spatter Code (BSC)" }
                            option { value: "map", "Multiply Add Permute (MAP)" }
                            option { value: "hrr", "Holographic Reduced Representation (HRR)" }
                            option { value: "vtb", "Vector-derived Transformation Binding (VTB)" }
                        }
                    }

                    div {
                        label { class: "mb-1 block text-sm font-medium text-slate-700",
                            "Embedding size: {dimensions()}"
                        }
                        input {
                            class: "w-full accent-slate-900",
                            r#type: "range",
                            min: "{MIN_DIMENSIONS}",
                            max: "{MAX_DIMENSIONS}",
                            step: "{DIMENSION_STEP}",
                            value: "{dimensions}",
                            oninput: move |event| on_dimensions_change.call(event.value()),
                        }
                    }

                    div {
                        label { class: "mb-1 block text-sm font-medium text-slate-700",
                            "Add element"
                        }
                        div { class: "flex flex-col gap-3",
                            input {
                                class: "w-full rounded-xl border border-slate-300 bg-white px-3 py-2 outline-none ring-offset-2 focus:border-slate-500 focus:ring-2 focus:ring-slate-400",
                                r#type: "text",
                                placeholder: "New element(s), for example: apple banana",
                                value: "{new_item}",
                                oninput: move |event| new_item.set(event.value()),
                            }
                            Button {
                                class: "w-full",
                                onclick: move |_| on_add_item.call(()),
                                "Add element"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn App() -> Element {
    let mut architecture = use_signal(|| Architecture::Bsc);
    let mut dimensions = use_signal(|| DEFAULT_DIMENSIONS);
    let mut storage = use_signal(|| AppStorage::new(Architecture::Bsc, DEFAULT_DIMENSIONS));
    let mut names = use_signal(Vec::<String>::new);
    let mut new_item = use_signal(String::new);
    let mut query = use_signal(String::new);

    let names_list = names();
    let query_result = storage.read().similarities(&names_list, &query());
    let query_error = query_result.as_ref().err().cloned();
    let similarities = query_result.unwrap_or_else(|_| vec![None; names_list.len()]);

    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }

        main { class: "min-h-screen bg-slate-50 px-4 py-10 text-slate-900",
            div { class: "mx-auto w-full max-w-6xl space-y-6",
                div { class: "grid items-stretch gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]",
                    ControlSidebar {
                        architecture,
                        dimensions,
                        new_item,
                        item_count: names_list.len(),
                        on_architecture_change: move |value: String| {
                            if let Ok(next_architecture) = value.parse::<Architecture>() {
                                if next_architecture != architecture() {
                                    architecture.set(next_architecture);
                                    storage.set(AppStorage::new(next_architecture, dimensions()));
                                    names.set(Vec::new());
                                }
                            }
                        },
                        on_dimensions_change: move |value: String| {
                            if let Ok(next_dimensions) = value.parse::<usize>() {
                                if next_dimensions != dimensions() {
                                    dimensions.set(next_dimensions);
                                    storage.set(AppStorage::new(architecture(), next_dimensions));
                                    names.set(Vec::new());
                                }
                            }
                        },
                        on_add_item: move |_| {
                            let raw_input = new_item();
                            let mut added_any = false;

                            for candidate in raw_input.split_whitespace() {
                                let candidate = candidate.to_string();
                                if candidate.is_empty() {
                                    continue;
                                }
                                if !storage.read().has(&candidate) {
                                    storage.write().push(candidate.clone());
                                    names.write().push(candidate);
                                    added_any = true;
                                }
                            }

                            if !added_any && raw_input.trim().is_empty() {
                                return;
                            }

                            new_item.set(String::new());
                        },
                    }

                    Card { class: "p-3",
                        CardHeader {
                            div { class: "flex items-center justify-between gap-3",
                                CardTitle { "Elements" }
                            }
                            CardDescription { "Added elements and their similarities to the query." }
                        }
                        CardContent {
                            if names.read().is_empty() {
                                Empty { class: "min-h-[24rem]",
                                    EmptyHeader {
                                        EmptyTitle { "No elements yet" }
                                        EmptyDescription { "Use the sidebar to add your first element." }
                                    }
                                }
                            } else {
                                div { class: "grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4",
                                    for (idx , name) in names_list.iter().enumerate() {
                                        {
                                            let similarity = similarities.get(idx).copied().unwrap_or(None);
                                            let (badge_label, badge_style) = similarity_badge(similarity);
                                            let tile_style = similarity_tile_style(similarity);
                                            rsx! {
                                                div {
                                                    key: "{name}",
                                                    class: "group flex min-h-24 flex-col gap-1.5 rounded-xl border p-2.5 shadow-sm transition-transform hover:-translate-y-0.5",
                                                    style: "{tile_style}",
                                                    div { class: "flex items-start justify-between gap-1.5",
                                                        Badge { class: "border-slate-200 bg-white/80 text-slate-700", "#{idx + 1}" }
                                                        span {
                                                            class: "inline-flex items-center justify-center rounded-full border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0",
                                                            style: "{badge_style}",
                                                            "{badge_label}"
                                                        }
                                                    }
                                                    div {
                                                        p { class: "break-all font-mono text-base font-semibold text-slate-900", "{name}" }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Card { class: "p-3",
                    CardHeader { class: "pb-2",
                        CardTitle { "Query" }
                        CardDescription { "Calculate the similarity between the added vectors and this query:" }
                    }
                    CardContent { class: "pt-0",
                        input {
                            class: "w-full rounded-xl border border-slate-300 bg-white px-3 py-2 outline-none ring-offset-2 focus:border-slate-500 focus:ring-2 focus:ring-slate-400",
                            r#type: "text",
                            placeholder: "Enter query expression",
                            value: "{query}",
                            oninput: move |event| query.set(event.value()),
                        }
                        if let Some(error) = query_error {
                            p { class: "mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 ring-1 ring-rose-200",
                                "{error}"
                            }
                        }
                    }
                }
            }
        }
    }
}
