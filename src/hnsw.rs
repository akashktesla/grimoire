#![allow(warnings)]
use crate::grimoire::Embedding;
use crate::hellindex::cosine_similarity;
use std::{collections::HashMap, hash::Hash};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };
use rand::Rng;
use std::fmt;
use std::fmt::Debug;
use pdf_extract::extract_text;
use crate::collections::KSArray;

pub fn main(){

    let text = extract_text("../src/The_Art_Of_War.pdf").unwrap();
    let chunk_size = 50;
    let chunks = chunking(text, chunk_size);

    let embedding_model_path:String = "/home/akash/.models/all-MiniLM-L6-v2".to_string();
    let level_probability:f64 = 0.36; //rounded from 0,36;
    let max_neighbours:usize = 16;
    let eq_construction :usize = 200;
    let eq_search:usize = 70;
    let mut engine:HnswEngine = HnswEngine::new(embedding_model_path, level_probability, eq_construction, eq_search,max_neighbours); 
    engine.load(chunks);
    println!("Engine: {:#?}",engine.level_nodes);
    let result = engine.traverse(&String::from("All warfare is based on deception"),&3);
    println!("Result: {:?}",result);
    // let result = engine.brute_force(String::from("All warfare is based on deception"),3);
    // println!("Result: {:?}",result);
    for i in result.nodes{
        println!("{:?}",engine.nodes.get(&i.node_id).unwrap().embedding.text);
    }
    //NOTE
    //bro in the loop ur taking only top as primary candidate but take all 3 and push it to the
    //KSArray
    
}

type NodeId = usize;
    
pub fn chunking(text: String, chunk_size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    for chunk in words.chunks(chunk_size) {
        let joined = chunk.join(" ");
        chunks.push(joined);
    }

    return chunks;
}


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
    level:i64, //level of the node
    neighbours: HashMap<i64, KSArray> // level - node_id
}


impl HnswNode{
    fn new(id:NodeId,embedding:Embedding,level:i64,neighbours:HashMap<i64,KSArray>)->HnswNode{
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
}



struct HnswEngine{
    entry_point: NodeId, //dynamically updated
    max_level:i64, // for tracking
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String,
    nodes: HashMap<NodeId,HnswNode>, //global pool of nodes
    level_nodes: HashMap<i64,Vec<NodeId>>,
    current_node_id: NodeId,
    level_probability:f64,
    max_neighbours:usize,
    eq_construction:usize,
    eq_search:usize,
}

impl Debug for HnswEngine{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HnswEngine: 
            entry_point: {:?}, 
            max_level: {},
            embedding_model_path: {},
            nodes: {:#?},
            level_nodes: {:?},
            current_node_id: {},
            level_probability: {},
            max_neighbours: {},
            eq_construction: {},
            eq_search: {}
            ",
            self.entry_point,
            self.max_level,
            self.embedding_model_path,
            self.nodes,
            self.level_nodes,
            self.current_node_id,
            self.level_probability,
            self.max_neighbours,
            self.eq_construction,
            self.eq_search)
    }

}



impl HnswEngine{
    fn new(embedding_model_path:String,
        level_probability:f64, eq_construction:usize, eq_search:usize, max_neighbours:usize) -> HnswEngine{


        let embedding_model = SentenceEmbeddingsBuilder
            ::local(&embedding_model_path)
            .create_model()
            .expect("couldn't create the model");

        let entry_point = 0;
        let max_level = -1;
        return HnswEngine{
            entry_point,
            max_level,
            embedding_model,
            embedding_model_path,
            nodes:HashMap::new(),
            level_nodes:HashMap::new(),
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
            let mut level = self.generate_level();
            // println!("level:{:?}",level);
            let node = HnswNode::new(self.current_node_id,embedding,level,HashMap::new());
            if level > self.max_level {
                self.entry_point = node.id;
                self.max_level = level;
            }
            //TODO Greedy search here to find neighbors
            let current_node_id = self.current_node_id.clone();
            self.nodes.insert(current_node_id,  node);

            while level>=0{
                //level nodes relationship 
                match self.level_nodes.get_mut(&level){
                    Some(node_list)=>{
                        node_list.push(current_node_id);
                    }
                    None=>{
                        self.level_nodes.insert(level,vec![current_node_id]);
                    }
                }
                self.update_neighbours_greedy(&current_node_id,&level);
                self.current_node_id = self.current_node_id+1;
                level -= 1;
            }
        }
    }

    fn update_neighbours_greedy(&mut self,node_id:&NodeId,level:&i64){
        let node =  self.nodes.get(node_id).unwrap();
        // println!("len: {:?}, eq_construction: {:?}",self.level_nodes.get(level).unwrap().len(),self.eq_construction);
        if self.level_nodes.get(level).unwrap().len()>self.eq_construction{
            //greedy
            let neighbours = self.traverse_construction(&node.embedding.text,level);
            // println!("neighbours: {:?}",neighbours);
            let node = self.nodes.get_mut(node_id).unwrap();
            for i in &neighbours.nodes{
                match node.neighbours.get_mut(&level){
                    Some(neighbour) => {
                        neighbour.insert_node(&i.node_id,&i.similarity);
                    }
                    none => {
                        node.neighbours.insert(*level,KSArray::new(self.max_neighbours));
                        node.neighbours.get_mut(&level).unwrap().insert_node(&i.node_id,&i.similarity);
                    }
                }
            }
            //updating neighbours
            // println!("similarities_vec:{:?}",similarities_vec);
            for i in &neighbours.nodes{ 
                match node.neighbours.get_mut(&level){
                    Some(neighbour) => {
                        neighbour.insert_node(&node_id,&i.similarity);
                    }
                    none => {
                        node.neighbours.insert(*level,KSArray::new(self.max_neighbours));
                        node.neighbours.get_mut(&level).unwrap().insert_node(&node_id,&i.similarity);
                    }
                } 
            }
        }

        else{//bruiteforce
            let embedding = &node.embedding.clone();
            let mut similarities_vec= Vec::new();
            let node_list  = self.level_nodes.get(&level).unwrap();
            //TODO: to implement a greedy approach when brute force > eq_construction
            for i in node_list{ 
                if node_id !=i{ //ignore calc similarity for same shit
                    let sec_embedding = self.nodes.get(i).unwrap().embedding.embedding.clone();
                    let similarity = cosine_similarity(&embedding.embedding, &sec_embedding);
                    similarities_vec.push((*i,similarity));
                }
            }
            //sorting by most similarity
            similarities_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            //inserting neighbors after sorting
            let b = 5;
            let max_neighbours = self.max_neighbours as usize;
            let node = self.nodes.get_mut(node_id).unwrap();
            for i in &similarities_vec[0..self.max_neighbours.min(similarities_vec.len())]{ 
                match node.neighbours.get_mut(&level){
                    Some(neighbour) => {
                        neighbour.insert_node(&i.0,&i.1);
                    }
                    None => {
                        node.neighbours.insert(*level,KSArray::new(self.max_neighbours));
                        node.neighbours.get_mut(&level).unwrap().insert_node(&i.0,&i.1);
                    }
                }
            }
            //updating neighbours
            // println!("similarities_vec:{:?}",similarities_vec);
            for i in &similarities_vec{ 
                let mut node = self.nodes.get_mut(&i.0).unwrap();
                match node.neighbours.get_mut(&level){
                    Some(neighbour) => {
                        neighbour.insert_node(&node_id,&i.1);
                    }
                    None => {
                        node.neighbours.insert(*level,KSArray::new(self.max_neighbours));
                        node.neighbours.get_mut(&level).unwrap().insert_node(&node_id,&i.1);
                    }
                }
            }
        }
    }

    fn brute_force(&self,user_query:String,k:usize)->KSArray{
        let mut returns = KSArray::new(k);
        //user_query embedding
        let uq_embedding = self.generate_embeddings_string(&user_query); 
        for i in self.nodes.values(){
            let similarity  = cosine_similarity(&uq_embedding.embedding,&i.embedding.embedding);
            returns.insert_node(&i.id,&similarity);

        }
        return returns;
    }

    fn traverse(&self,user_query:&String,k:&usize)->KSArray{
        return self.traverse_core(user_query,k,&self.eq_search);
    }

    fn traverse_construction(&self,user_query:&String,level:&i64,)->KSArray{
        let mut returns = KSArray::new(self.max_neighbours);
        let uq_embedding = self.generate_embeddings_string(user_query); 
        let mut eqs = self.eq_construction;
        let mut prime_candidate = self.nodes.get(&self.level_nodes.get(&level).unwrap()[0]).unwrap();
        let mut highest_similarity=0.;
        while eqs > 0 {
            eqs-=1;
            let mut vec_similarity = Vec::new();
            match  prime_candidate.neighbours.get(&level) {
                Some(prime_candidate_neighbours)=>{
                    for i in &prime_candidate_neighbours.nodes{
                        let neighbour = self.nodes.get(&i.node_id).unwrap();
                        let similarity  = cosine_similarity(&uq_embedding.embedding,&neighbour.embedding.embedding);
                        vec_similarity.push((i.node_id,similarity));
                        returns.insert_node(&i.node_id,&similarity);
                    }
                    //sorting by similarity
                    vec_similarity.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
                    //Travel to that node
                    if vec_similarity[0].1 > highest_similarity{
                        prime_candidate = self.nodes.get(&vec_similarity[0].0).unwrap();
                        highest_similarity = vec_similarity[0].1;
                    }
                }
                None=>{
                    eqs = 0;
                }
            }
        }
        return returns;
    }

    fn traverse_core(&self,user_query:&String,k:&usize,eq_search:&usize)->KSArray{
        let mut returns = KSArray::new(*k);
        let mut eqs = *eq_search;
        //user_query embedding
        let uq_embedding = self.generate_embeddings_string(user_query); 
        //compare with entry point's neighbour and travel
        let entry_point = self.nodes.get(&self.entry_point).unwrap();
        let mut level:i64 = entry_point.level;
        let mut prime_candidate = entry_point;
        let mut highest_similarity:f32 = 0.;
        while level >= 0 && eqs > 0 {
            eqs-=1;

            let mut vec_similarity = Vec::new();
            match  prime_candidate.neighbours.get(&level) {
                Some(prime_candidate_neighbours)=>{
                    for i in &prime_candidate_neighbours.nodes{
                        let neighbour = self.nodes.get(&i.node_id).unwrap();
                        let similarity  = cosine_similarity(&uq_embedding.embedding,&neighbour.embedding.embedding);
                        vec_similarity.push((i.node_id,similarity));
                        returns.insert_node(&i.node_id,&similarity);
                    }
                    //sorting by similarity
                    vec_similarity.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
                    //Travel to that node
                    if vec_similarity[0].1 > highest_similarity{
                        prime_candidate = self.nodes.get(&vec_similarity[0].0).unwrap();
                        highest_similarity = vec_similarity[0].1;
                    }
                    else{
                        level-=1;
                    }
                    // println!("Iteration: eqs: {:?},level: {:?}",eqs,level);
                }
                None=>{
                    level -=1;

                }
            }
        }
        return returns;
    }

    fn generate_level(&self) -> i64 {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        let scale = 1.0 / self.level_probability.ln(); // scale = 1 / ln(1 / prob)
        let level = (-r.ln() * scale).floor() as i64;
        return level.abs() - 1;
    }

    fn generate_embeddings_string(&self,payload:&String)->Embedding{
        let embedding = self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
        return Embedding::new(payload.clone(),embedding);
    }

}


