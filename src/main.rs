#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, };
use ::grimoire::embeddings;
use ::grimoire::grimoire;
use ::grimoire::hellindex;

fn main() {
    hellindex::main();
    // grimoire::main();
}

