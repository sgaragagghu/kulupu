// SPDX-License-Identifier: GPL-3.0-or-later
// This file is part of Kulupu.
//
// Copyright (c) 2020 Wei Tang.
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

mod v1;
mod v2;

pub use self::v1::{ComputeV1, SealV1};
pub use self::v2::{ComputeV2, SealV2};
pub use randomx::Config;

use log::info;
use codec::{Encode, Decode};
use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use sp_core::H256;
use lazy_static::lazy_static;
use lru_cache::LruCache;
use once_cell::sync::OnceCell;
use kulupu_randomx as randomx;
use kulupu_primitives::Difficulty;

// Using this macro, it is possible to have statics that require code to be executed at
// runtime in order to be initialized. This includes anything requiring heap allocations, 
// like vectors or hash maps, as well as anything that requires function calls to be computed.
lazy_static! {
	static ref FULL_SHARED_CACHES: Arc<Mutex<LruCache<H256, Arc<randomx::FullCache>>>> =
		Arc::new(Mutex::new(LruCache::new(2)));
	static ref LIGHT_SHARED_CACHES: Arc<Mutex<LruCache<H256, Arc<randomx::LightCache>>>> =
		Arc::new(Mutex::new(LruCache::new(3)));
}

// A thread local storage key which owns its contents.
thread_local! { // TODO understand RefCell
	static FULL_MACHINE: RefCell<Option<(H256, randomx::FullVM)>> = RefCell::new(None);
	static LIGHT_MACHINE: RefCell<Option<(H256, randomx::LightVM)>> = RefCell::new(None);
}

// Shareable mutable containers.

// Rust memory safety is based on this rule: Given an object T, it is only possible to have one of the following:

//    Having several immutable references (&T) to the object (also known as aliasing).
//    Having one mutable reference (&mut T) to the object (also known as mutability).

// A cell which can be written to only once.
static GLOBAL_CONFIG: OnceCell<Config> = OnceCell::new();
static DEFAULT_CONFIG: Config = Config::new();

pub fn global_config() -> &'static Config {
	GLOBAL_CONFIG.get().unwrap_or(&DEFAULT_CONFIG)
}

pub fn set_global_config(config: Config) -> Result<(), Config> {
	GLOBAL_CONFIG.set(config)
}

#[derive(Clone, Copy, PartialEq, Eq, Encode, Decode, Debug)]
pub enum ComputeMode {
	Sync,
	Mining,
}

#[derive(Clone, Copy, PartialEq, Eq, Encode, Decode, Debug)]
pub enum Loop<R> {
	Continue,
	Break(R),
}

#[derive(Clone, PartialEq, Eq, Encode, Decode, Debug)]
pub struct Calculation {
	pub pre_hash: H256,
	pub difficulty: Difficulty,
	pub nonce: H256,
}

fn need_new_vm<M: randomx::WithCacheMode>(
	key_hash: &H256,
	machine: &RefCell<Option<(H256, randomx::VM<M>)>>,
) -> bool {
	let ms = machine.borrow(); // TODO is it an array of machines or just one ?

	let need_new_vm = ms.as_ref().map(|(mkey_hash, _)| {
		mkey_hash != key_hash // if the key is not up to date i guess..
	}).unwrap_or(true); // if none so there isnt... it returns true! so make a new one, ok !

	need_new_vm
}

// TODO understand what's happening here. I guess it's mining, it's the interface between rust and c++ randomX
// AFAIK should return the machine... ?!
fn loop_raw_with_cache<M: randomx::WithCacheMode, FPre, I, FValidate, R>(
	key_hash: &H256,
	machine: &RefCell<Option<(H256, randomx::VM<M>)>>,
	shared_caches: &Arc<Mutex<LruCache<H256, Arc<randomx::Cache<M>>>>>,
	mut f_pre: FPre,
	f_validate: FValidate,
	round: usize,
) -> Option<R> where
	FPre: FnMut() -> (Vec<u8>, I),
	FValidate: Fn(H256, I) -> Loop<Option<R>>,
{
	if need_new_vm(key_hash, machine) { // if key hash is new or there is no machine
		let mut ms = machine.borrow_mut(); // borrow and mut i guess like &mut &...

		// Panics if the value is a None with a custom panic message provided by msg.
		// pub fn lock(&self) -> LockResult<MutexGuard<'_, T>>
		let mut shared_caches = shared_caches.lock().expect("Mutex poisioned"); // poisoned means the lock has been release
											// because of a panic...

// pub fn get_mut(&mut self) -> &mut Tⓘ
// Returns a mutable reference to the underlying data.
// This call borrows Cell mutably (at compile-time) which guarantees that we possess the only reference

		if let Some(cache) = shared_caches.get_mut(key_hash) {
			*ms = Some((*key_hash, randomx::VM::new(cache.clone(), global_config()))); // finally creating the VM here.. it seems
								// as you can see below it needs cache ptr and cache datasetptr
// pub struct VM<M: WithCacheMode> {
//	_cache: Arc<Cache<M>>,
//	ptr: *mut sys::randomx_vm,
// }
// pub struct Cache<M: WithCacheMode> {
//	cache_ptr: *mut sys::randomx_cache,
//	dataset_ptr: Option<*mut sys::randomx_dataset>,
//	_marker: PhantomData<M>,
// }
// pub fn new(cache: Arc<Cache<M>>, config: &Config) -> Self {
// 	let flags = M::randomx_flags(config);
//
//	let ptr = unsafe {
//		sys::randomx_create_vm( 
//			flags,
//			cache.cache_ptr,
//			cache.dataset_ptr.unwrap_or(std::ptr::null_mut()),
//		)
//	};
//	Self { _cache: cache, ptr }
// }
		} else {
			info!(
				target: "kulupu-randomx",
				"At block boundary, generating new RandomX {} cache with key hash {} ...",
				M::description(),
				key_hash,
			);
			let cache = Arc::new(randomx::Cache::new(&key_hash[..], global_config()));
			shared_caches.insert(*key_hash, cache.clone());
			*ms = Some((*key_hash, randomx::VM::new(cache, global_config())));
		}
	}

	let mut ms = machine.borrow_mut();

	let ret = ms.as_mut()
		.map(|(mkey_hash, vm)| {
			assert_eq!(mkey_hash, key_hash,
					   "Condition failed checking cached key_hash. This is a bug");

			let mut ret = None;

			match round {
				0 => (),
				1 => {
					let (pre, int) = f_pre();
					let hash = H256::from(vm.calculate(&pre[..]));
					let validate = f_validate(hash, int);

					match validate {
						Loop::Continue => (),
						Loop::Break(b) => {
							ret = b;
						},
					}
				},
				_ => {
					let (prev_pre, mut prev_int) = f_pre();
					let mut vmn = vm.begin(&prev_pre[..]);

					for _ in 1..round {
						let (pre, int) = f_pre();
						let prev_hash = H256::from(vmn.next(&pre[..]));
						let prev_validate = f_validate(prev_hash, prev_int);

						prev_int = int;

						match prev_validate {
							Loop::Continue => (),
							Loop::Break(b) => {
								ret = b;
								break
							},
						}
					}

					let prev_hash = H256::from(vmn.finish());
					let prev_validate = f_validate(prev_hash, prev_int);

					match prev_validate {
						Loop::Continue => (),
						Loop::Break(b) => {
							ret = b;
						},
					}
				}
			}

			ret
		})
		.expect("Local MACHINES always set to Some above; qed");

	ret
}

pub fn loop_raw<FPre, I, FValidate, R>(
	key_hash: &H256,
	mode: ComputeMode, // mode...
	f_pre: FPre,
	f_validate: FValidate,
	round: usize,
) -> Option<R> where
	FPre: FnMut() -> (Vec<u8>, I),
	FValidate: Fn(H256, I) -> Loop<Option<R>>,
{
	match mode {
		ComputeMode::Mining => // mining...

// pub fn with<F, R>(&'static self, f: F) -> R where
//     F: FnOnce(&T) -> R, 
// Acquires a reference to the value in this TLS key.
// This will lazily initialize the value if this thread has not referenced this key yet.
		
			FULL_MACHINE.with(|machine| { // if there wasnt it-s initializing it now! Why don-t we check the result here ?
				loop_raw_with_cache::<randomx::WithFullCacheMode, _, _, _, _>( // BUT this is a function... se the result of this function
												// should kinda be a machine
					key_hash, // the famouse kayhash... it's just the hash of an 'old' block (still must understand the meaning...)
					machine, // closure object
					&FULL_SHARED_CACHES, // like global variable cache for every threads
					f_pre, // more or less is the ComputeV1 struct. i guess it's a callback to 
						// create the computeV1. Definitely... it's callef f_!
					f_validate, // callback to decide if wehave to break mining ()cause we've got above the target difficulty
						   // or continue...
					round, // just a mining parameter
				)
			}),
		ComputeMode::Sync => { 
			let full_ret = FULL_MACHINE.with(|machine| {
				if !need_new_vm::<randomx::WithFullCacheMode>(key_hash, machine) {
					Ok(loop_raw_with_cache::<randomx::WithFullCacheMode, _, _, _, _>(
						key_hash,
						machine,
						&FULL_SHARED_CACHES,
						f_pre,
						f_validate,
						round,
					))
				} else {
					Err((f_pre, f_validate))
				}
			});

			match full_ret {
				Ok(ret) => ret,
				Err((f_pre, f_validate)) => {
					LIGHT_MACHINE.with(|machine| {
						loop_raw_with_cache::<randomx::WithLightCacheMode, _, _, _, _>(
							key_hash,
							machine,
							&LIGHT_SHARED_CACHES,
							f_pre,
							f_validate,
							round,
						)
					})
				}
			}
		},
	}
}

pub fn compute<T: Encode>(key_hash: &H256, input: &T, mode: ComputeMode) -> H256 {
	loop_raw(
		key_hash,
		mode,
		|| (input.encode(), ()),
		|hash, ()| Loop::Break(Some(hash)),
		1,
	).expect("Loop break always returns Some; qed")
}

#[cfg(test)]
mod tests {
	use super::*;
	use sp_core::{H256, U256};

	#[test]
	fn randomx_len() {
		assert_eq!(randomx::HASH_SIZE, 32);
	}

	#[test]
	fn randomx_collision() {
		let mut compute = ComputeV1 {
			key_hash: H256::from([210, 164, 216, 149, 3, 68, 116, 1, 239, 110, 111, 48, 180, 102, 53, 180, 91, 84, 242, 90, 101, 12, 71, 70, 75, 83, 17, 249, 214, 253, 71, 89]),
			pre_hash: H256::default(),
			difficulty: U256::default(),
			nonce: H256::default(),
		};
		let hash1 = compute.clone().seal_and_work(ComputeMode::Sync);
		U256::one().to_big_endian(&mut compute.nonce[..]);
		let hash2 = compute.seal_and_work(ComputeMode::Sync);
		assert!(hash1.1 != hash2.1);

		let mut compute2 = ComputeV2 {
			key_hash: H256::from([210, 164, 216, 149, 3, 68, 116, 1, 239, 110, 111, 48, 180, 102, 53, 180, 91, 84, 242, 90, 101, 12, 71, 70, 75, 83, 17, 249, 214, 253, 71, 89]),
			pre_hash: H256::default(),
			difficulty: U256::default(),
			nonce: H256::default(),
		};
		let hash3 = compute2.clone().seal_and_work(Default::default(), ComputeMode::Sync);
		U256::one().to_big_endian(&mut compute2.nonce[..]);
		let hash4 = compute2.seal_and_work(Default::default(), ComputeMode::Sync);
		assert!(hash3.1 != hash4.1);
		assert!(hash1.1 != hash3.1);
		assert!(hash2.1 != hash4.1);
	}
}
