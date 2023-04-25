use std::{convert::Infallible, fs::File, io::Write};

use ggml_rs_bindings::{loader::LoadError, Model};
use rand::SeedableRng;

extern crate llama_ggml;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let loc = &args[1];

    let file = File::open(loc)
        .map_err(|e| LoadError::OpenFileFailed {
            source: e,
            path: loc.to_owned(),
        })
        .unwrap();

    println!(" >>> Loading model from {loc}...");
    let now = std::time::Instant::now();

    let model = TryInto::<llama_ggml::Llama>::try_into(&file).unwrap();

    println!(" >>> Model loaded in {} ms. {:?}\n", now.elapsed().as_millis(), model.hyperparameters());

    let mut session = model.start_session();
    let res = session.inference_from_prompt::<Infallible>(
        &model,
        &Default::default(),
        "The best kind of wine is ",
        None,
        &mut rand::rngs::StdRng::from_entropy(),
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();

            Ok(())
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}
