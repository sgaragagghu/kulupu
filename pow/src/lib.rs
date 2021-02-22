// SPDX-License-Identifier: GPL-3.0-or-later
// This file is part of Kulupu.
//
// Copyright (c) 2019-2020 Wei Tang.
//
// Kulupu is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Kulupu is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Kulupu. If not, see <http://www.gnu.org/licenses/>.

pub mod compute;
pub mod weak_sub;

use std::{sync::Arc, time::{Duration, Instant}};
use parking_lot::Mutex;
use codec::{Encode, Decode};
use sp_core::{U256, H256, blake2_256};
use sp_api::ProvideRuntimeApi;
use sp_runtime::generic::BlockId;
use sp_runtime::traits::{
	Block as BlockT, Header as HeaderT, UniqueSaturatedInto,
};
use sp_consensus_pow::{Seal as RawSeal, DifficultyApi};
use sc_consensus_pow::PowAlgorithm;
use sc_client_api::{blockchain::HeaderBackend, backend::AuxStore};
use sc_keystore::LocalKeystore;
use kulupu_primitives::{Difficulty, AlgorithmApi};
use rand::{SeedableRng, thread_rng, rngs::SmallRng};
use log::*;

use crate::compute::{ComputeV1, ComputeV2, SealV1, SealV2, ComputeMode};

pub mod app {
	use sp_application_crypto::{app_crypto, sr25519};
	use sp_core::crypto::KeyTypeId;

	pub const ID: KeyTypeId = KeyTypeId(*b"klp2");
	app_crypto!(sr25519, ID);
}

/// Checks whether the given hash is above difficulty.
pub fn is_valid_hash(hash: &H256, difficulty: Difficulty) -> bool {
	let num_hash = U256::from(&hash[..]); // i guess normal conversion from a hash type to a unsigned normal number...
	//  returns (a * b) mod 2^N (N in this case must be 256..)
	let (_, overflowed) = num_hash.overflowing_mul(difficulty); // _ is the multiplication and overflowed is a bool...
	// it makes sense, higher difficulty means i have to find a smaller hash!
	
	!overflowed // if has not overflowed is valid...
}

pub fn key_hash<B, C>(
	client: &C,
	parent: &BlockId<B>
) -> Result<H256, sc_consensus_pow::Error<B>> where
	B: BlockT<Hash=H256>,
	C: HeaderBackend<B>, // Blockchain database header backend. Does not perform any validation.
{
	const PERIOD: u64 = 4096; // ~2.8 days
	const OFFSET: u64 = 128;  // 2 hours
	
// The header is omething which fulfills the abstract idea of a Substrate header. 
// It has types for a Number, a Hash and a Hashing. It provides access to an extrinsics_root, 
// state_root and parent_hash, as well as a digest and a block number.


	// pub fn header(&self, id: BlockId<Block>) -> Result<Option<Block::Header>>
	let parent_header = client.header(*parent) // get the header of the parent block
		.map_err(|e| sc_consensus_pow::Error::Environment( // remapping every error to this error sc_consensus_pow::Error::Environment
			format!("Client execution error: {:?}", e)
		))? // if Ok then unwrap, if err then return!
		// Transforms the Option<T> into a Result<T, E>, mapping Some(v) to Ok(v) and None to Err(err).
		.ok_or(sc_consensus_pow::Error::Environment( // throwing this error bsically, in this case it will return..
			"Parent header not found".to_string()
		))?; // if some thn unwrap, if none then return!
	// Just like Into except that if the source value is too big to fit into the destination type then it'll saturate the destination.
	// But consume self! (the `caller`)
	// As you know in rust, you have the ownership principles, which define that a function that owns the variable, 
	// will destroy it at the end. So consumption is this process, of owning something and dropping it.
	let parent_number = UniqueSaturatedInto::<u64>::unique_saturated_into(*parent_header.number()); // Number is.. the header number. But not sure what is it..
											// could it be the block number ?  i still dont know at this point 

	let mut key_number = parent_number.saturating_sub(parent_number % PERIOD); // parent number - (parent_numberModuloPeriod)...
	if parent_number.saturating_sub(key_number) < OFFSET { 	// basically it is  (parent number -  (parent_number - (parent_numberModPeriod))) 
								// that`s basically parent_numberModPeriod < offset
		key_number = key_number.saturating_sub(PERIOD); // parent number - (parent_numberModuloPeriod) - (period)
	}

	let mut current = parent_header;
	while UniqueSaturatedInto::<u64>::unique_saturated_into(*current.number()) != key_number {
		current = client.header(BlockId::Hash(*current.parent_hash())) // i guess going back through the chain`s headers
			.map_err(|e| sc_consensus_pow::Error::Environment(
				format!("Client execution error: {:?}", e)
			))?
			.ok_or(sc_consensus_pow::Error::Environment(
				format!("Block with hash {:?} not found", current.hash())
			))?;
	}

	Ok(current.hash())
}

pub enum RandomXAlgorithmVersion {
	V1,
	V2,
}

// A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically Reference Counted'.
// The type Arc<T> provides shared ownership of a value of type T, 
// allocated in the heap. Invoking clone on Arc produces a new Arc instance, 
// which points to the same allocation on the heap as the source Arc, 
// while increasing a reference count. When the last Arc pointer to a given allocation is destroyed, 
// the value stored in that allocation (often referred to as "inner value") is also dropped.
pub struct RandomXAlgorithm<C> {
	client: Arc<C>,
}

impl<C> RandomXAlgorithm<C> {
	pub fn new(client: Arc<C>) -> Self {
		Self {
			client,
		}
	}
}

impl<C> Clone for RandomXAlgorithm<C> {
	fn clone(&self) -> Self {
		Self {
			client: self.client.clone(),
		}
	}
}

impl<B: BlockT<Hash=H256>, C> PowAlgorithm<B> for RandomXAlgorithm<C> where
	C: HeaderBackend<B> + AuxStore + ProvideRuntimeApi<B>,
	C::Api: DifficultyApi<B, Difficulty> + AlgorithmApi<B>,
{
	type Difficulty = Difficulty; // associated type
// With associated types, we don’t need to annotate types because we can’t implement 
// a trait on a type multiple times. In Listing 19-12 with the definition that uses associated types, 
// we can only choose what the type of Item will be once, because there can only be one impl Iterator for Counter. 
// We don’t have to specify that we want an iterator of u32 values everywhere that we call next on Counter.

	// getting the difficult of the `last?` block
	fn difficulty(&self, parent: H256) -> Result<Difficulty, sc_consensus_pow::Error<B>> {
		let difficulty = self.client.runtime_api().difficulty(&BlockId::Hash(parent)) // TODO understand
			.map_err(|e| sc_consensus_pow::Error::Environment( // mapping every error to sc_consensus_pow::Error::Environment
				format!("Fetching difficulty from runtime failed: {:?}", e)
			));

		difficulty
	}

	fn break_tie( // TODO understsand... where is it used ?
		&self,
		own_seal: &RawSeal,
		new_seal: &RawSeal,
	) -> bool {
		// blake2 just is doing the has of the seal...
		blake2_256(&own_seal[..]) > blake2_256(&new_seal[..])
	}

	fn verify( // TODO understand, perhaps it is verifying block to import them... i guess
		&self,
		parent: &BlockId<B>,
		pre_hash: &H256,
		pre_digest: Option<&[u8]>,
		seal: &RawSeal,
		difficulty: Difficulty,
	) -> Result<bool, sc_consensus_pow::Error<B>> {
		let version_raw = self.client.runtime_api().identifier(parent)
			.map_err(|e| sc_consensus_pow::Error::Environment(
				format!("Fetching identifier from runtime failed: {:?}", e))
			)?;

		let version = match version_raw {
			kulupu_primitives::ALGORITHM_IDENTIFIER_V1 => RandomXAlgorithmVersion::V1,
			kulupu_primitives::ALGORITHM_IDENTIFIER_V2 => RandomXAlgorithmVersion::V2,
			_ => return Err(sc_consensus_pow::Error::<B>::Other(
				"Unknown algorithm identifier".to_string(),
			)),
		};

		let key_hash = key_hash(self.client.as_ref(), parent)?;

		match version {
			RandomXAlgorithmVersion::V1 => {
				let seal = match SealV1::decode(&mut &seal[..]) {
					Ok(seal) => seal,
					Err(_) => return Ok(false),
				};

				let compute = ComputeV1 {
					key_hash,
					difficulty,
					pre_hash: *pre_hash,
					nonce: seal.nonce,
				};

				// No pre-digest check is needed for V1 algorithm.

				let (computed_seal, computed_work) = compute.seal_and_work(ComputeMode::Sync);

				if computed_seal != seal {
					return Ok(false)
				}

				if !is_valid_hash(&computed_work, difficulty) {
					return Ok(false)
				}

				Ok(true)
			},
			RandomXAlgorithmVersion::V2 => {
				let seal = match SealV2::decode(&mut &seal[..]) {
					Ok(seal) => seal,
					Err(_) => return Ok(false),
				};

				let compute = ComputeV2 {
					key_hash,
					difficulty,
					pre_hash: *pre_hash,
					nonce: seal.nonce,
				};

				let pre_digest = match pre_digest {
					Some(pre_digest) => pre_digest,
					None => return Ok(false),
				};

				let author = match app::Public::decode(&mut &pre_digest[..]) {
					Ok(author) => author,
					Err(_) => return Ok(false),
				};

				if !compute.verify(&seal.signature, &author) {
					return Ok(false)
				}

				let (computed_seal, computed_work) = compute.seal_and_work(
					seal.signature.clone(),
					ComputeMode::Sync,
				);

				if computed_seal != seal {
					return Ok(false)
				}

				if !is_valid_hash(&computed_work, difficulty) {
					return Ok(false)
				}

				Ok(true)
			},
		}
	}
}

pub struct Stats {
	last_clear: Instant,
	last_display: Instant,
	round: u32,
}

impl Stats {
	pub fn new() -> Stats {
		Self {
			last_clear: Instant::now(),
			last_display: Instant::now(),
			round: 0,
		}
	}
}

pub fn mine<B, C>(
	client: &C,
	keystore: &LocalKeystore,
	parent: &BlockId<B>,
	pre_hash: &H256,
	pre_digest: Option<&[u8]>,
	difficulty: Difficulty,
	round: u32,
	stats: &Arc<Mutex<Stats>>,
) -> Result<Option<RawSeal>, sc_consensus_pow::Error<B>> where
	B: BlockT<Hash=H256>,
	C: HeaderBackend<B> + AuxStore + ProvideRuntimeApi<B>,
	C::Api: DifficultyApi<B, Difficulty> + AlgorithmApi<B>,
{
	let version_raw = client.runtime_api().identifier(parent)
		.map_err(|e| sc_consensus_pow::Error::Environment(
			format!("Fetching identifier from runtime failed: {:?}", e))
		)?;

	let version = match version_raw {
		kulupu_primitives::ALGORITHM_IDENTIFIER_V1 => RandomXAlgorithmVersion::V1,
		kulupu_primitives::ALGORITHM_IDENTIFIER_V2 => RandomXAlgorithmVersion::V2,
		_ => return Err(sc_consensus_pow::Error::<B>::Other(
			"Unknown algorithm identifier".to_string()
		)),
	};

	let mut rng = SmallRng::from_rng(&mut thread_rng())
		.map_err(|e| sc_consensus_pow::Error::Environment(
			format!("Initialize RNG failed for mining: {:?}", e)
		))?;
	let key_hash = key_hash(client, parent)?;

	let pre_digest = pre_digest.ok_or(sc_consensus_pow::Error::<B>::Other(
		"Unable to mine: pre-digest not set".to_string(),
	))?;

	let author = app::Public::decode(&mut &pre_digest[..]).map_err(|_| {
		sc_consensus_pow::Error::<B>::Other(
			"Unable to mine: author pre-digest decoding failed".to_string(),
		)
	})?;

	let pair = keystore.key_pair::<app::Pair>(
		&author,
	).map_err(|_| sc_consensus_pow::Error::<B>::Other(
		"Unable to mine: fetch pair from author failed".to_string(),
	))?;

	let maybe_seal = match version {
		RandomXAlgorithmVersion::V1 => {
			compute::loop_raw(
				&key_hash,
				ComputeMode::Mining,
				|| {
					let nonce = H256::random_using(&mut rng);

					let compute = ComputeV1 {
						key_hash,
						difficulty,
						pre_hash: *pre_hash,
						nonce,
					};

					(compute.input().encode(), compute)
				},
				|work, compute| {
					if is_valid_hash(&work, compute.difficulty) {
						let seal = compute.seal();
						compute::Loop::Break(Some(seal.encode()))
					} else {
						compute::Loop::Continue
					}
				},
				round as usize,
			)
		},
		RandomXAlgorithmVersion::V2 => {
			compute::loop_raw(
				&key_hash,
				ComputeMode::Mining,
				|| {
					let nonce = H256::random_using(&mut rng);

					let compute = ComputeV2 {
						key_hash,
						difficulty,
						pre_hash: *pre_hash,
						nonce,
					};

					let signature = compute.sign(&pair);

					(compute.input(signature.clone()).encode(), (compute, signature))
				},
				|work, (compute, signature)| {
					if is_valid_hash(&work, difficulty) {
						let seal = compute.seal(signature);
						compute::Loop::Break(Some(seal.encode()))
					} else {
						compute::Loop::Continue
					}
				},
				round as usize,
			)
		},
	};

	let now = Instant::now();

	let maybe_display = {
		let mut stats = stats.lock();
		let since_last_clear = now.checked_duration_since(stats.last_clear);
		let since_last_display = now.checked_duration_since(stats.last_display);

		if let (Some(since_last_clear), Some(since_last_display)) =
			(since_last_clear, since_last_display)
		{
			let mut ret = None;

			stats.round += round;
			let duration = since_last_clear;

			let clear = duration >= Duration::new(600, 0);
			let display = (clear || since_last_display >= Duration::new(2, 0)) && duration.as_secs() > 0;

			if display {
				stats.last_display = now;
				ret = Some((duration, stats.round));
			}

			if clear {
				stats.last_clear = now;
				stats.round = 0;
			}

			ret
		} else {
			warn!(
				target: "kulupu-pow",
				"Calculating duration failed, the system time may have changed and the hashrate calculation may be temporarily inaccurate."
			);

			None
		}
	};

	if let Some((duration, round)) = maybe_display {
		let hashrate = round / duration.as_secs() as u32;
		let network_hashrate = difficulty / U256::from(60);

		if hashrate == 0 {
			info!(
				target: "kulupu-pow",
				"Local hashrate: {} H/s, network hashrate: {} H/s",
				hashrate,
				network_hashrate,
			);
		} else {
			let every: u32 = (network_hashrate / U256::from(hashrate)).unique_saturated_into();
			let every_duration = Duration::new(60, 0) * every;
			info!(
				target: "kulupu-pow",
				"Local hashrate: {} H/s, network hashrate: {} H/s, expected one block every {} ({} blocks)",
				hashrate,
				network_hashrate,
				humantime::format_duration(every_duration).to_string(),
				every,
			);
		}
	}

	Ok(maybe_seal)
}
