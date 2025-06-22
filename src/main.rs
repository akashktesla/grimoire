#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, };
use ::grimoire::grimoire;
use ::grimoire::hellindex;
use ::grimoire::hnsw;

fn main() {
    // hellindex::main();
    // grimoire::main();
    hnsw::main();
}

