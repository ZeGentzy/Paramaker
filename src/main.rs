//! Badly written and disorginized, paramaker is an example of exactly what you
//! shouldn't do when programming in Rust.
//!
//! For an explanation for how it works, refer to the README.
//!
//! To change the inputs, look at `main()`.
//!
//! Best compiled with `cargo run --release`.

#![feature(nll, no_panic_pow)]

//extern crate primal;
extern crate rayon;

//use primal::Sieve;
use rayon::prelude::*;
use std::{borrow::Borrow, ops::Range};

#[derive(Clone, Copy)]
enum Token {
	Number(f64),
	Divide,
	Multiply,
	Add,
	Subtract,
	BracketStart,
	BracketEnd,
	Power,
}

#[derive(Clone, Copy, Debug)]
enum Direction {
	Before,
	After,
}

#[derive(Debug)]
struct TokenData {
	pre_chunks:         Option<Vec<Range<usize>>>,
	post_chunks:        Option<Vec<Range<usize>>>,
	pre_divide_factor:  Option<usize>,
	post_divide_factor: Option<usize>,
}

#[derive(Debug)]
struct DerivedTokenData {
	pre_chunks:  Option<Range<usize>>,
	post_chunks: Option<Range<usize>>,
}

impl std::fmt::Debug for Token {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		use Token::*;
		let sym = match self {
			Number(n) => n.to_string(),
			Divide => " / ".to_string(),
			Multiply => " * ".to_string(),
			Add => " + ".to_string(),
			Subtract => " - ".to_string(),
			BracketStart => "(".to_string(),
			BracketEnd => ")".to_string(),
			Power => " ^ ".to_string(),
		};
		write!(f, "{}", sym)
	}
}

impl Token {
	fn do_op(&self, left: f64, right: f64) -> f64 {
		use Token::*;
		match self {
			BracketStart | BracketEnd | Number(_) => panic!(),
			Divide => left / right,
			Multiply => left * right,
			Add => left + right,
			Subtract => left - right,
			Power => left.powf(right),
		}
	}

	fn to_str<T: Borrow<[Self]>>(tokens: &T) -> String {
		let mut ret: String = "".to_string();
		for t in tokens.borrow() {
			use Token::*;
			match t {
				Number(n) => ret += &n.to_string(),
				Divide => ret += " / ",
				Multiply => ret += " * ",
				Add => ret += " + ",
				Subtract => ret += " - ",
				BracketStart => ret += "(",
				BracketEnd => ret += ")",
				Power => ret += " ^ ",
			}
		}

		ret
	}

	fn to_str_t<O, T: Borrow<[(Self, O)]>>(tokens: &T) -> String
	where
		O: std::fmt::Debug,
	{
		let mut ret: String = "".to_string();
		ret.push_str(&Self::to_str(
			&tokens.borrow().iter().map(|(t, _)| *t).collect::<Vec<_>>(),
		));
		ret += "\n";

		for (t, o) in tokens.borrow() {
			ret += &format!("{:<10}: {:?}\n", Self::to_str(&[*t]), o);
		}

		ret
	}

	/// Returns the depth of a token.
	///
	/// Eg:
	/// tokens:  (1) + 2 + (3 + (3))
	/// Returns: 010 0 0 0 01 1 1210
	fn depth<T: Borrow<[Self]>>(tokens: &T, index: usize) -> usize {
		use Token::*;

		let mut cur = 0;
		let mut depth = 0;
		let tokens = tokens.borrow();

		while cur <= index {
			match tokens[cur] {
				BracketStart => depth += 1,
				BracketEnd => depth -= 1,
				_ => (),
			}
			cur += 1;
		}

		if let BracketStart = tokens[index] {
			depth - 1
		} else {
			depth
		}
	}

	/// Chunks around an operation. Must be at the same depth.
	/// Eg:
	/// tokens: (3 + 2) + (3) + 2 + 2 + 8 + (2 + 2) + (1)
	///                               ^- index = 13
	/// Returns: ([(0..5), (6..9), (10..11), (12..13)], [(14..15), (16..21),
	/// (22..25)])
	///
	/// Eg:
	/// tokens: (1 + (2 + 3 + 5 + 4) + 1 + (1 + 2))
	///                     ^- index = 7
	/// Returns: ([(4..5), (6..7)], [(8..9), (10..11)])
	fn chunks_around_oper<T: Borrow<[Self]>>(
		tokens: &T,
		index: usize,
	) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
		use Token::*;

		let tokens = tokens.borrow();

		let this_depth = Self::depth(&tokens, index);

		let mut start = index;
		let mut end = index;

		while start > 0 && Self::depth(&tokens, start - 1) >= this_depth {
			start -= 1
		}

		while end < (tokens.len() - 1)
			&& Self::depth(&tokens, end + 1) >= this_depth
		{
			end += 1
		}

		let mut ret = vec![];
		let mut pre = None;
		let mut offset = 0;
		let mut chunk_start = None;

		for t in &tokens[start..end + 1] {
			// TODO: Possible optimization: track our depth when chunk_start !=
			// None, to avoid recalling Self::depth.
			match (t, &mut chunk_start) {
				(BracketStart, None) => {
					chunk_start = Some(start + offset);
				},
				(Number(_), None) => {
					ret.push((start + offset)..(start + offset + 1));
					if start + offset + 1 == index {
						if let None = pre {
							pre = Some(ret);
							ret = vec![]
						}
					}
				},
				(BracketEnd, chunk_start) => {
					let depth = Self::depth(&tokens, start + offset);
					if depth == this_depth {
						ret.push(chunk_start.unwrap()..(start + offset + 1));
						*chunk_start = None;
						if start + offset + 1 == index {
							if let None = pre {
								pre = Some(ret);
								ret = vec![]
							}
						}
					}
				},
				_ => (),
			}

			offset += 1;
		}

		assert_eq!(chunk_start, None);

		(pre.unwrap(), ret)
	}

	fn solve<T: Borrow<[Self]>>(tokens: &T) -> f64 {
		use Token::*;
		let mut values = vec![0.];
		let mut lasts: Vec<Option<Token>> = vec![None];

		for t in tokens.borrow() {
			match (t, lasts.last_mut().unwrap()) {
				(Number(n), None) => *values.last_mut().unwrap() = *n,
				(Number(n), Some(op)) => {
					*values.last_mut().unwrap() =
						op.do_op(*values.last().unwrap(), *n)
				},

				(BracketStart, _) => {
					values.push(0.);
					lasts.push(None);
				},
				(BracketEnd, _) => {
					let lvalue = values.pop().unwrap();
					lasts.pop();
					let last = lasts.last_mut().unwrap();
					match last {
						None => {
							*values.last_mut().unwrap() =
								values.last().unwrap() + lvalue
						},
						Some(op) => {
							*values.last_mut().unwrap() =
								op.do_op(*values.last().unwrap(), lvalue)
						},
					}
				},

				(next, last) => *last = Some(*next),
			}
		}

		values[0]
	}

	/// Operations are compatible if their relative order of operation in
	/// irrelivent.
	///
	/// The rules are as follows:
	///
	/// Power operations always have to compute the effects of having brackets
	/// before and after the symbol.
	///
	/// Division and subtraction operations always have to compute the effects
	/// of having brackets after the operation.
	///
	/// Division and multiplication operations only have to compute the effects
	/// of having brackets before the operation, if we have an addition or
	/// subtraction operation before them. Multiplication operations also apply
	/// this same rule for operations after them, too.
	///
	/// Subtraction and addition operations only have to compute the effects of
	/// having brackets before the operation, if we have an multiplication or
	/// division operation before them. Addition operations also apply this same
	/// rule for operations after them, too.
	fn is_compatible<T: Borrow<[Self]>>(
		tokens: &T,
		dir: Direction,
		chunks: &(Vec<Range<usize>>, Vec<Range<usize>>),
	) -> bool {
		let index = chunks.0.last().unwrap().end;

		let tokens = tokens.borrow();
		let chunks = match dir {
			Direction::Before => &chunks.0,
			Direction::After => &chunks.1,
		};
		let token = tokens[index];

		use Token::*;
		match (token, dir) {
			(Power, _) => return false,
			(Divide, Direction::After) | (Subtract, Direction::After) => {
				return false
			},
			_ => (),
		}

		for chunk in &chunks[0..chunks.len() - 1] {
			match (token, tokens[chunk.end]) {
				(Divide, Add) | (Divide, Subtract) => return false,
				(Subtract, Divide) | (Subtract, Multiply) => return false,
				(Multiply, Add) | (Multiply, Subtract) => return false,
				(Add, Divide) | (Add, Multiply) => return false,
				_ => (),
			}
		}

		true
	}

	/*fn primeify<T: Borrow<[Self]>>(tokens: &T) -> Vec<Token> {
        use Token::*;
        let mut ret = vec![];
        let tokens = tokens.borrow();

        for token in tokens {
            match token {
                Number(n) => {
                    ret.push(BracketStart);
                    // TODO: Cache prime sieve
                    // TODO: Consider not converting to a float and back.
                    let sieve = Sieve::new(f64::sqrt(*n as f64) as f64 + 1);
                    let primes = sieve.factor(*n).unwrap();

                    let bracket_inner = primes.len() != 1;

                    for (base, power) in primes {
                        if bracket_inner { ret.push(BracketStart) }
                        ret.push(Number(base));
                        ret.push(Power);
                        ret.push(Number(power));
                        if bracket_inner { ret.push(BracketEnd) }
                        ret.push(Multiply);
                    }
                    ret.pop(); // Remove extra Multiply at end
                    ret.push(BracketEnd);
                }
                _ => ret.push(*token),
            }
        }

        ret
    }*/

	fn derive_state<T: Borrow<[(Self, Option<TokenData>)]>>(
		tokens: &T,
		state: usize,
	) -> Vec<(Self, Option<DerivedTokenData>)> {
		let tokens = tokens.borrow();
		tokens
			.par_iter()
			.map(|(token, data)| {
				let dd = data.as_ref().map(|data| DerivedTokenData {
					pre_chunks:  if let Some(ref c) = data.pre_chunks {
						Some(
							c[(state / data.pre_divide_factor.unwrap())
							      % c.len()]
								.start..c.last().unwrap().end,
						)
					} else {
						None
					},
					post_chunks: if let Some(ref c) = data.post_chunks {
						Some(
							c[0].start
								..c[(state / data.post_divide_factor.unwrap())
									    % c.len()]
									.end,
						)
					} else {
						None
					},
				});

				(*token, dd)
			})
			.collect()
	}

	fn insert_brackets<T: Borrow<[(Self, Option<DerivedTokenData>)]>>(
		tokens: &T,
	) -> Vec<Self> {
		let tokens = tokens.borrow();
		let (pre_chunks, post_chunks): (Vec<_>, Vec<_>) = tokens
			.par_iter()
			.filter_map(|(_, dtd)| {
				dtd.as_ref().map(|dtd| {
					(dtd.pre_chunks.clone(), dtd.post_chunks.clone())
				})
			})
			.unzip();

		let (mut br_starts, mut br_ends): (Vec<_>, Vec<_>) = pre_chunks
			.par_iter()
			.chain(post_chunks.par_iter())
			.filter_map(|br| br.clone().map(|br| (br.start, br.end)))
			.unzip();

		br_starts.sort();
		br_ends.sort();

		let mut tokens: Vec<_> = tokens.par_iter().map(|(t, _)| *t).collect();
		while !br_starts.is_empty() || !br_ends.is_empty() {
			let br_start = br_starts.last().map(|i| *i);
			let br_end = br_ends.last().map(|i| *i);

			use Token::{BracketEnd, BracketStart};
			match (br_start, br_end) {
				(None, None) => unreachable!(),
				(Some(i), None) => {
					br_starts.pop();
					tokens.insert(i, BracketStart);
				},
				(None, Some(i)) => {
					br_ends.pop();
					tokens.insert(i, BracketEnd);
				},
				(Some(i1), Some(i2)) => {
					if i1 >= i2 {
						br_starts.pop();
						tokens.insert(i1, BracketStart);
					} else {
						br_ends.pop();
						tokens.insert(i2, BracketEnd);
					}
				},
			}
		}

		tokens
	}
}

fn main() {
	use Token::*;
	let eq_left: Vec<Token> = vec![
		BracketStart,
		Number(123.),
		Add,
		Number(2323.),
		Multiply,
		Number(1135374.),
		Divide,
		BracketStart,
		Number(2323.),
		Multiply,
		Number(232.),
		Add,
		Number(122.),
		Add,
		BracketStart,
		Number(343548.),
		Divide,
		Number(2.),
		Add,
		Number(10.),
		BracketEnd,
		BracketEnd,
		BracketEnd,
		Add,
		Number(230.),
		Divide,
		Number(2.),
	];

	let eq_right: Vec<Token> = vec![Number(4999.), Divide, Number(2.)];

	//let eq_left = Token::primeify(&eq_left);
	//let eq_right = Token::primeify(&eq_right);

	let eq_right_solved = Token::solve(&eq_right);

	println!("Left hand side: {}", Token::to_str(&eq_left));
	println!(
		"Right hand side: {} = {}\n",
		Token::to_str(&eq_right),
		eq_right_solved
	);

	let eq_left: Vec<_> = eq_left
		.par_iter()
		.enumerate()
		.map(|(i, token)| match token {
			Number(n) => (Number(*n), None),
			BracketStart => (BracketStart, None),
			BracketEnd => (BracketEnd, None),
			op => {
				let chunks = Token::chunks_around_oper(&eq_left, i);
				let do_before =
					!Token::is_compatible(&eq_left, Direction::Before, &chunks);
				let do_after =
					!Token::is_compatible(&eq_left, Direction::After, &chunks);

				if do_after || do_before {
					let td = TokenData {
						pre_chunks:         if do_before {
							Some(chunks.0)
						} else {
							None
						},
						post_chunks:        if do_after {
							Some(chunks.1)
						} else {
							None
						},
						pre_divide_factor:  None,
						post_divide_factor: None,
					};
					(*op, Some(td))
				} else {
					(*op, None)
				}
			},
		})
		.collect();

	let mut next_divide_factor = 1;
	let eq_left: Vec<_> = eq_left
		.into_iter()
		.map(|(t, data)| match data {
			Some(mut data) => {
				let pre_divide_factor = match data.pre_chunks {
					None => None,
					Some(ref c) => {
						let ret = Some(next_divide_factor);
						next_divide_factor *= c.len();
						ret
					},
				};
				let post_divide_factor = match data.post_chunks {
					None => None,
					Some(ref c) => {
						let ret = Some(next_divide_factor);
						next_divide_factor *= c.len();
						ret
					},
				};

				data.pre_divide_factor = pre_divide_factor;
				data.post_divide_factor = post_divide_factor;

				(t, Some(data))
			},
			None => (t, None),
		})
		.collect();

	println!("{}", Token::to_str_t(&eq_left));
	println!(
		"Max possible attempt{}: {}",
		if next_divide_factor == 1 { "" } else { "s" },
		next_divide_factor
	);

	let found = (0..next_divide_factor).into_par_iter().find_any(|&s| {
		let state = Token::derive_state(&eq_left, s);
		let state = Token::insert_brackets(&state);
		let state_solved = Token::solve(&state);
		if state_solved == eq_right_solved {
			true
		} else {
			false
		}
	});

	if let Some(s) = found {
		let state = Token::derive_state(&eq_left, s);
		let state = Token::insert_brackets(&state);
		println!("Found state number {}.", s + 1,);
		println!(
			"{} = {} = {}",
			Token::to_str(&state),
			eq_right_solved,
			Token::to_str(&eq_right)
		);
	} else {
		println!("No matches.")
	}
}
