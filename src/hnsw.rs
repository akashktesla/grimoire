#![allow(warnings)]
use crate::grimoire::Embedding;
use crate::hellindex::cosine_similarity;
use std::collections::HashMap;
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };
use rand::Rng;

pub fn main(){

    let sentences = vec![
        String::from("Akash is a god"),
        String::from("Life is all about ups and downs"),
        String::from("I like rust language it has memory safety"),
        String::from("Rust is a good language"),
        String::from("Akash is a great guy"),
        String::from("God doesn't exist"),
    ];
    let max_level = 5;
     
    let embedding_model_path:String = "/home/akash/.models/all-MiniLM-L6-v2".to_string();
    let level_probability:f64 = 0.4; //rounded from 0,36;
    let max_neighbours:u32 = 5;
    let eq_construction :u32 = 200;
    let eq_search:u32 = 70;
    let mut engine:HnswEngine = HnswEngine::new(max_level, embedding_model_path, level_probability, eq_construction, eq_search,max_neighbours); 
    engine.load(sentences);
    
}

type NodeId = u32;
    
#[derive(Debug, Clone)]
struct Neighbour {
    node_id: NodeId,
    similarity: f32,
}

impl Neighbour{
    fn new(node_id:NodeId,similarity:f32)->Neighbour{
        return Neighbour{
            node_id,
            similarity
        }

    }
}


#[derive(Clone, Debug)]
struct HnswNode{
    id:NodeId,
    embedding: Embedding, 
    level:u32, //level of the node
    neighbours: HashMap<u32, Vec<Neighbour>> // level - node_id
}

impl HnswNode{
    fn new(id:NodeId,embedding:Embedding,level:u32,neighbours:HashMap<u32,Vec<Neighbour>>)->HnswNode{
        return HnswNode{
            id,
            embedding,
            level,
            neighbours
        }
    }

    fn new_empty()->HnswNode{
        return HnswNode{
            id:0,
            embedding:Embedding::new_empty(),
            level:0,
            neighbours:HashMap::new()
        }

    }

    fn insert_neighbour(&mut self, level:u32, node_id:NodeId, similarity:f32, max_neighbours:u32){
        match self.neighbours.get_mut(&level){
            Some(neighbours)=>{
                if neighbours.len() as u32 > max_neighbours { //aldready full
                    if similarity < neighbours.last().unwrap().similarity { //optimization 
                        return; //break the function
                    }
                    neighbours.push(Neighbour::new(node_id,similarity));
                    neighbours.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
                    neighbours.pop();
                }
                else{
                    neighbours.push(Neighbour::new(node_id,similarity));
                    neighbours.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
                }
            },
            None=>{
                self.neighbours.insert(level,vec![Neighbour::new(level,similarity)]);
            }
        }
    }
}

struct HnswEngine{
    entry_point: HnswNode, //dynamically updated
    max_level:u32, // for tracking
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String,
    nodes: HashMap<NodeId,HnswNode>, //global pool of nodes
    current_node_id: NodeId,
    level_probability:f64,
    max_neighbours:u32,
    eq_construction:u32,
    eq_search:u32,
}

impl HnswEngine{
    fn new(max_level:u32, embedding_model_path:String,
        level_probability:f64, eq_construction:u32, eq_search:u32, max_neighbours:u32) -> HnswEngine{


        let embedding_model = SentenceEmbeddingsBuilder
            ::local(&embedding_model_path)
            .create_model()
            .expect("couldn't create the model");

        let entry_point = HnswNode::new_empty();
        return HnswEngine{
            entry_point,
            max_level,
            embedding_model,
            embedding_model_path,
            nodes:HashMap::new(),
            current_node_id:0,
            level_probability,
            max_neighbours,
            eq_construction,
            eq_search,
        }
    }

    fn load(&mut self, chunks:Vec<String>){
        for i in chunks{
            let embedding = self.generate_embeddings_string(&i);
            let level = self.generate_level();
            let node = HnswNode::new(self.current_node_id,embedding,level,HashMap::new());
            if level > self.max_level {
                self.entry_point = node.clone();
            }
            //TODO Greedy search here to find neighbors
            let current_node_id = self.current_node_id.clone();
            self.nodes.insert(current_node_id,  node);
            self.update_neighbours_greedy(&current_node_id,level);
            self.current_node_id = self.current_node_id+1
        }
    }

    fn update_neighbours_greedy(&mut self,node_id:&NodeId,level:u32){
        //TODO rework this shit to consider all levels bro
        // println!("node values: {:?}",self.nodes.values());
        let node = self.nodes.get_mut(node_id).unwrap();
        let embedding = &self.nodes.get(node_id).unwrap().embedding;
        for i in self.nodes.values(){
            let similarity = cosine_similarity(&embedding.embedding, &i.embedding.embedding);
            node.insert_neighbour(level) //fix this shit
            println!("i:{},similarity: {}",i.embedding.text,similarity);
        }
    }

    fn traverse(user_query:String,){

    }

    fn generate_level(&self) -> u32 {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        let scale = 1.0 / self.level_probability.ln(); // scale = 1 / ln(1 / prob)
        let level = (-r.ln() * scale).floor() as u32;
        level.min(self.max_level) // cap to max_level
    }

    fn generate_embeddings_string(&self,payload:&String)->Embedding{
        let embedding = self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
        return Embedding::new(payload.clone(),embedding);
      }

}
