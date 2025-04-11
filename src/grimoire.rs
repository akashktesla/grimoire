#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use crate::embeddings::{generate_embeddings_string,generate_embeddings_vec};
use crate::hellindex::{generate_metadata};
use rustc_hash::FxHashMap;

pub fn main() {
    let path = "/home/akash/projects/grimoire/src/test.grm".to_string();
    let payload = vec!["Akash likes cooking".to_string(),"Ram does all night coding".to_string()];
    let mut vdb = Grimoire::new(path,payload,10);
    vdb.save_db();
    vdb.similarity_search("cooking".to_string());
    // vdb.load_db();
    // vdb.insert_string("Akash also loves fucking with others".to_string());
    println!("vdb: {:?}",vdb);
}


#[derive(Encode, Decode, PartialEq, Debug)]
struct Embedding{
    text: String,
    embedding: Vec<f32>
}
impl Embedding{
    fn new(text:String, embedding:Vec<f32>)->Self{
        return Embedding{
            text,
            embedding,
        }
    }
}


#[derive(Encode, Decode, PartialEq, Debug)]
struct Grimoire{
    path: String,
    db: FxHashMap<Vec<Vec<i32>>,Vec<Embedding>>, //chunk_number -> Embedding
    rcn:FxHashMap<i32,Vec<Vec<Vec<i32>>>>, //Rank -> Chunk number lookup^
    chunk_size:i32,
    rank_list: Vec<i32>
}

impl Grimoire{

    fn new(path: String, payload: Vec<String>,chunk_size:i32) -> Self {
        let embeddings:Vec<Vec<f32>> = generate_embeddings_vec(payload.clone());
        let mut db:FxHashMap<Vec<Vec<i32>>,Vec<Embedding>> = FxHashMap::default();
        let mut rcn:FxHashMap<i32, Vec<Vec<Vec<i32>>>> = FxHashMap::default();

        for (embedding, text) in embeddings.into_iter().zip(payload){
            let (chunk_number,rank) = generate_metadata(&embedding,&chunk_size);
            db.entry(chunk_number.clone())
                .or_default()
                .push(Embedding::new(text,embedding));

            rcn.entry(rank)
                .or_default()
                .push(chunk_number);
            rank_list
        }

        Self {
            path,
            db,
            rcn,
            chunk_size,
            rank_list
        }
    }

    
    fn save_db(&self){
        let config = config::standard();
        let encoded = bincode::encode_to_vec(self,config).unwrap();
        write(self.path.clone(),&encoded)
            .expect(&format!("Unablel to write the file on path: {}",self.path));
    }

    fn load_db(&mut self){
        let contents = read(self.path.clone()).unwrap();
        let config = config::standard();
        let (loaded,_):(Grimoire,usize) = bincode::decode_from_slice(&contents,config).unwrap();
        *self = loaded;
        println!("contents: {:?}", self);
    }

    fn insert_string(&mut self,payload:String){
        //TODO later
    }

    fn similarity_search(&self,text:String){
        let embeddings = generate_embeddings_string(&text);
        let (user_chunk_id,user_rank) = generate_metadata(&embeddings,&self.chunk_size);
        println!("user_chunk_id: {:?}, user_rank: {:?}",user_chunk_id,user_rank);

    }
















}
