use std::{error::Error, prelude::v1::Result};

use burn::{
    backend::Wgpu,
    nn::{Embedding, EmbeddingConfig},
    tensor::{Int, Tensor},
};
use tokenizers::Tokenizer;

type Backend = Wgpu;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());

    let vocab_size = tokenizer.get_vocab_size(false);
    let embeddings_dimension = 1024;
    println!(
        "vocab_size=[{:}],embeddings_dimension[{:}]",
        vocab_size, embeddings_dimension
    );
    let embedding_config = EmbeddingConfig::new(vocab_size, embeddings_dimension);
    let embed = embedding_config.init::<Backend>();

    let token_ids = encoding
        .get_ids()
        .into_iter()
        .map(|t| t.clone() as i32)
        .collect::<Vec<i32>>();
    let tokens: Tensor<Wgpu, 1, Int> = Tensor::from_ints(token_ids.as_slice());
    let tokens = tokens.unsqueeze();
    let forward_pass = embed.forward(tokens);

    println!("{:?}", forward_pass.shape());
    Ok(())
    // let tensor = Tensor::<Backend,2>::from([[2]]);
}
