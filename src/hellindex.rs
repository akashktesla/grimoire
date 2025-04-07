#![allow(warnings)]
pub fn main(){
    let embedding = vec![0.7,0.32,0.43,0.64];
    let chunk_number = generate_chunk_number(embedding,10);
    println!("chunk_number: {:?}",chunk_number);
}

fn generate_chunk_number(embedding: Vec<f32>,chunk_size:i32)->(Vec<i32>,i64){
   let chunk_number:Vec<i32> = embedding
       .into_iter()
       .map(|i| (i*(chunk_size as f32)) as i32)
       .collect();

    let base_chunk_number: i64 = chunk_number
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &val)| (val as i64) * (chunk_size as i64).pow(i as u32))
        .sum();

    return (chunk_number,base_chunk_number)
}
