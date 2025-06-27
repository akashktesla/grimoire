#![allow(warnings)]
pub fn main(){
    let embedding = vec![0.000000002,0.0032,0.00043,0.14];
    let num = 0.002;
    let exponent = get_exponent(&num);
    println!("Exponent: {:?}",exponent);
    // let chunk_number = generate_metadata(embedding,10);
    // println!("chunk_number: {:?}",chunk_number);
    // let chunk_number_1 = vec![4,6,7,3];
    // let chunk_number_2 = vec![4,5,8,3];
    // let diff = calculate_difference(chunk_number_1,chunk_number_2);
    // println!("diff: {:?}",diff);

}

//lookup table
const POW10: [f32; 77] = [
    1e-38, 1e-37, 1e-36, 1e-35, 1e-34, 1e-33, 1e-32, 1e-31, 1e-30,
    1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20,
    1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
    1e-9,  1e-8,  1e-7,  1e-6,  1e-5,  1e-4,  1e-3,  1e-2,  1e-1,  1e0,
    1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,   1e10,
    1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,  1e20,
    1e21,  1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,  1e30,
    1e31,  1e32,  1e33,  1e34,  1e35,  1e36,  1e37,  1e38
];

#[inline(always)]
fn exp10i(exp: i32) -> f32 {
    // Clamp to safe range for f32 
    let clamped = exp.clamp(-38, 38);
    POW10[(clamped + 38) as usize]
}
fn get_exponent(x: &f32) -> (f32, i32) {
    if *x == 0.0 {
        return (0.0, 0);
    }
    let sign = if *x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let bits = abs_x.to_bits();
    let exp2 = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut exp10 = (exp2 as f32 * 0.30103).floor() as i32;

    let mut mantissa = abs_x / exp10i(exp10);

    // Normalize to [1.0, 10.0)
    if mantissa >= 10.0 {
        mantissa /= 10.0;
        exp10 += 1;
    } else if mantissa < 1.0 {
        mantissa *= 10.0;
        exp10 -= 1;
    }

    return (sign * mantissa, exp10);
}

pub fn generate_metadata(embedding: &Vec<f32>,chunk_size:&i32)->(Vec<Vec<i32>>,i32){
    let mut chunk_vector = vec![vec![],vec![]];
    for &i in embedding{
        let (mantissa, exp) = get_exponent(&i);
        let scaled = ((mantissa + 10.0) / 20.0) * (*chunk_size as f32);
        let chunk_number = scaled.floor() as i32;
        chunk_vector[0].push(chunk_number);
        chunk_vector[1].push(exp.abs());
    }
    let pe = positional_encoding(chunk_vector[0].len());
    let rank:f32 = chunk_vector[0]
        .iter()
        .zip(pe.iter())
        .map(|(val, weight)| *val as f32 * weight)
        .sum(); 

    return (chunk_vector,rank as i32)
}

fn positional_encoding(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let angle = i as f32 / 10000_f32.powf(i as f32 / dim as f32);
            if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            }
        })
        .collect()
}

pub fn calculate_difference(mcn1:Vec<i32>,mcn2:Vec<i32>,ecn1:Vec<i32>,ecn2:Vec<i32>)->i32{
    let mantissa_diff:i32 =  mcn1
        .into_iter()
        .zip(mcn2)
        .map(|(a,b)|(a-b).abs())
        .sum();
    let mantissa_diff:i32 =  ecn1
            .into_iter()
            .zip(ecn2)
            .map(|(a,b)|(a-b).abs())
            .sum();



    return 0;
}

pub fn cosine_similarity(a:&Vec<f32>, b:&Vec<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0; // Avoid division by zero
    }

    dot_product / (norm_a * norm_b)
}
