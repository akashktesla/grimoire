#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, };
use ::grimoire::grimoire;
use ::grimoire::hellindex;
use ::grimoire::hnsw;
use pdf_extract::extract_text;

fn main() {
    // hellindex::main();
    // grimoire::main();
    // hnsw::main();
    let text = extract_text("../src/The_Art_Of_War.pdf").unwrap();
    let chunk_size = 
    println!("text: {:?}",text);

}

