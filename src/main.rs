#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, };

fn main() {
    println!("valzkai ae");
    let path = "/home/akash/projects/grimoire/src/test.grm".to_string();
    let mut vdb = Grimoire::new(path);
    // vdb.save_db();
    vdb.load_db();
    println!("vdb: {:?}",vdb);
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct Grimoire{
    path: String,
    db: Vec<(String,Vec<f32>)>
}

impl Grimoire{
    fn new(path:String,payload:Vec<String>)->Self{
        return Grimoire{
            path,
            db : vec![(vec![1.,2.,3.],String::from("Hi I'm akash"))]
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

    fn generate_embeddings(&mut self, payload:Vec<&str>){
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()
            .expect("Couldn't Load the embedding model");
        // Encode the sentences into embeddings.
        let embeddings = model.encode(&mut payload).expect("Failed to encode the string");
        debug!("sucessfully created embeddings");
    }

}







