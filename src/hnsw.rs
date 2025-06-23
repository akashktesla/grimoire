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

type NodeId = usize;

struct HnswNode{
    id:NodeId,
    embedding: Embedding, 
    level:i32, //level of the node
    levels: HashMap<i32, Vec<NodeId>> // level - node_id
}

struct HnswEngine{
    entry_point: HnswNode, //dynamically updated
    max_level:i32, // for tracking
    nodes: HashMap<NodeId,HnswNode> //global pool of nodes
}





