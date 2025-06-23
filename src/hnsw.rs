#![allow(warnings)]
use crate::grimoire::Embedding;
use std::collections::HashMap;

pub fn main(){

    let sentences = vec![
        String::from("Akash is a god"),
        String::from("Life is all about ups and downs"),
        String::from("I like rust language it has memory safety"),
        String::from("Rust is a good language"),
        String::from("Akash is a great guy"),
        String::from("God doesn't exist"),
    ];
}


struct HnswNode{
    id:usize,
    embedding: Embedding,
    levels: HashMap<i32, Vec<HnswNode>>
}





