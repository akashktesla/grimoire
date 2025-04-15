#![allow(warnings)]
use std::fs::{write, read};
use std::io::{self, Read};
use std::fs;
use crate::hellindex::{generate_metadata};
use rustc_hash::FxHashMap; use std::collections::BTreeMap;
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };

pub fn main() {
    let path = "/home/akash/projects/grimoire/src/test.grm".to_string(); let payload = vec!["Akash likes cooking".to_string(),"Ram does all night coding".to_string()];
    let emodel_path = "/home/akash/.models/all-MiniLM-L6-v2".to_string();
    let mut vdb = Grimoire::new(path,payload,10,emodel_path);
    // vdb.save_db();
    // vdb.load_db();
    vdb.similarity_search("cooking".to_string());
    // vdb.insert_string("Akash also loves fucking with others".to_string());
    // println!("vdb: {:?}",vdb);
}

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

struct Grimoire{
    path: String,
    db: FxHashMap<Vec<Vec<i32>>,Vec<Embedding>>, //chunk_number -> Embedding
    rcn:BTreeMap<i32,Vec<Vec<Vec<i32>>>>, //Rank -> Chunk number lookup^
    chunk_size:i32,
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String
}

impl Grimoire{

    fn new(path: String,payload: Vec<String>,chunk_size:i32,embedding_model_path:String) -> Self {
        let embedding_model = SentenceEmbeddingsBuilder
            ::local(&embedding_model_path)
            .create_model()
            .expect("couldn't create the model");

        let embeddings:Vec<Vec<f32>> = embedding_model.encode(&payload).expect("Failed to encode the string");
        let mut db:FxHashMap<Vec<Vec<i32>>,Vec<Embedding>> = FxHashMap::default();
        let mut rcn:BTreeMap<i32, Vec<Vec<Vec<i32>>>> = BTreeMap::new();

        for (embedding, text) in embeddings.into_iter().zip(payload){ 
            let (chunk_number,rank) = generate_metadata(&embedding,&chunk_size);
            db.entry(chunk_number.clone())
                .or_default()
                .push(Embedding::new(text,embedding));

            rcn.entry(rank)
                .or_default()
                .push(chunk_number);
        }

        return Self {
            path,
            db,
            rcn,
            chunk_size,
            embedding_model,
            embedding_model_path
        };
    }

    fn generate_embeddings_vec(&self,payload:Vec<String>)->Vec<Vec<f32>>{
        return self.embedding_model.encode(&payload).expect("Failed to encode the string");
    }

    fn generate_embeddings_string(&self,payload:&String)->Vec<f32>{
        return self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
    }

    fn serialize(&self)->Vec<u8>{
        let path_bytes = self.path.as_bytes().to_vec();
        let len_path_bytes = path_bytes.len().to_le_bytes();
        let mut data_to_save = Vec::new();
        // println!("path bytes: {:?}, len: {:08b}",path_bytes, len_path_bytes);
        data_to_save.extend(len_path_bytes);
        data_to_save.extend(path_bytes);
        // println!("{:?}",to_save);
        return data_to_save;

    }
    fn save_db(&self){
        let data_to_save = self.serialize();
        fs::write(&self.path, data_to_save).expect("Unable to write the file");
    }

    fn deserialize(&self){
        let mut file_data = Vec::new();
        let mut file = fs::File::open(&self.path).expect("Unable to open the file");
        file.read_to_end(&mut file_data).expect("Unable to read the file");

        let length_bytes = &file_data[0..8]; 
        let path_len = u64::from_le_bytes(length_bytes.try_into().expect("Invalid length bytes"));
        println!("path_len:{:?}",path_len);

        // Step 3: Deserialize the path bytes using the length we read
        let path_bytes = &file_data[8..(8 + path_len as usize)]; // Get the path bytes based on length
        println!("path_bytes:{:?}",path_bytes);


        // Step 4: Convert the path bytes back to a String
        let path = String::from_utf8(path_bytes.to_vec()).expect("Invalid UTF-8 sequence");
        println!("path: {:?}",path);

    }

    fn load_db(&self){
        self.deserialize();
    }

    fn insert_string(&mut self,payload:String){
        //TODO later
    }

    fn similarity_search(&self,text:String){
        let embeddings = self.generate_embeddings_string(&text);
        let (user_chunk_id,user_rank) = generate_metadata(&embeddings,&self.chunk_size);
        // println!("user_chunk_id: {:?}, user_rank: {:?}",user_chunk_id,user_rank);
        match self.rcn.get(&user_rank){
            Some(chunk_id)=>{
                let embedding = self.db.get(&chunk_id[0]).unwrap();
                println!("Embedding: {:?}",embedding[0].text)

            }
            None=>{
                match self.rcn.range(..user_rank).next_back(){
                    Some((rank,chunk_id))=>{
                        // println!("val: {:?}",val);
                        let embedding = self.db.get(&chunk_id[0]).unwrap();
                        println!("Embedding: {:?}",embedding[0].text)
                    }
                    None=>{
                        println!("No value found");
                    }

                }
            }
        }
    }

}
