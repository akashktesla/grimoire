#![allow(warnings)]
use crate::grimoire::Embedding;
use std::collections::HashMap;
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };

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
impl HnswNode{
    fn new(id:NodeId,embedding:Embedding,level:i32,levels:HashMap<i32,Vec<NodeId>>)->HnswNode{
        return HnswNode{
            id,
            embedding,
            level,
            levels
        }

    }

}

struct HnswEngine{
    entry_point: HnswNode, //dynamically updated
    max_level:i32, // for tracking
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String,
    nodes: HashMap<NodeId,HnswNode>, //global pool of nodes
    current_node_id: NodeId
                                   
}

impl HnswEngine{
    fn new(max_level:i32,embedding_model:SentenceEmbeddingsModel,embedding_model_path:String){
    }
    fn load(&mut self,chunks:Vec<String>){
        for i in chunks{
            let embedding = self.generate_embeddings_string(&i);
            let level = self.generate_level();
            let node = HnswNode::new(self.current_node_id,embedding,level,HashMap::new());
            self.nodes.insert(self.current_node_id,  node);
        }
    }
    fn generate_level(&self)->i32{
        return 0
    }
    fn generate_embeddings_string(&self,payload:&String)->Embedding{
        let embedding = self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
        return Embedding::new(payload.clone(),embedding);
      }

}



