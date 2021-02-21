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

//! Difficulty adjustment module.

#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Encode, Decode};
use sp_std::cmp::{min, max};
use sp_core::U256;
use sp_runtime::traits::UniqueSaturatedInto;
use sp_timestamp::OnTimestampSet;
use frame_support::{decl_storage, decl_module, traits::Get};
use kulupu_primitives::{
	DIFFICULTY_ADJUST_WINDOW, DIFFICULTY_DAMP_FACTOR, CLAMP_FACTOR,
	MIN_DIFFICULTY, MAX_DIFFICULTY, Difficulty,
};

#[derive(Encode, Decode, Clone, Copy, Eq, PartialEq, Debug)]
pub struct DifficultyAndTimestamp<M> {
	pub difficulty: Difficulty,
	pub timestamp: M,
}

/// Move value linearly toward a goal
pub fn damp(actual: u128, goal: u128, damp_factor: u128) -> u128 {
	(actual + (damp_factor - 1) * goal) / damp_factor
}

/// limit value to be within some factor from a goal
pub fn clamp(actual: u128, goal: u128, clamp_factor: u128) -> u128 {
	max(goal / clamp_factor, min(actual, goal * clamp_factor))
}

pub trait Config: pallet_timestamp::Config { //configuring another pallet...!
	/// Target block time in millseconds.
	type TargetBlockTime: Get<Self::Moment>; //Moment is defined there, in pallet_timestamp
	// ^^^^^^ it`s new.... there isnt in normal pallet_timestamp
}

decl_storage! {
	trait Store for Module<T: Config> as Difficulty {
		/// Past difficulties and timestamps, from earliest to latest.
		PastDifficultiesAndTimestamps:
		[Option<DifficultyAndTimestamp<T::Moment>>; 60] // just array with these two types for each element (so its a matrix)
			= [None; DIFFICULTY_ADJUST_WINDOW as usize];
		/// Current difficulty.
		pub CurrentDifficulty get(fn difficulty) build(|config: &GenesisConfig| {
			config.initial_difficulty
		}): Difficulty;
		/// Initial difficulty.
		pub InitialDifficulty config(initial_difficulty): Difficulty;
	}
}

decl_module! {
	// instead of calls here there is just a const...
	// IT IS a call! it is a const! COOL.
	pub struct Module<T: Config> for enum Call where origin: T::Origin {
		/// Target block time in milliseconds.
		const TargetBlockTime: T::Moment = T::TargetBlockTime::get();
	}
}

// Something which can be notified when the timestamp is set. Set this to () if not needed. 
// i guess the timestamp of the block... so when mined...?
impl<T: Config> OnTimestampSet<T::Moment> for Module<T> {
	fn on_timestamp_set(now: T::Moment) { // i guess now is the timestamp
		// Just like Into except that if the source value is too big to fit into the destination type then it'll saturate the destination.
		let block_time = UniqueSaturatedInto::<u128>::unique_saturated_into(T::TargetBlockTime::get()); // safe downcast basically
		let block_time_window = DIFFICULTY_ADJUST_WINDOW as u128 * block_time;

		// getting from storage
		let mut data = PastDifficultiesAndTimestamps::<T>::get();

		// shifting the array losing the oldest. it is kinda making space for a new ... element?
		for i in 1..data.len() {
			data[i - 1] = data[i];
		}

		// instatianing the new element with the current block i guess, since we ar in the ontimestampset function..
		data[data.len() - 1] = Some(DifficultyAndTimestamp {
			timestamp: now,
			difficulty: Self::difficulty(),
		});

		let mut ts_delta = 0;
		for i in 1..(DIFFICULTY_ADJUST_WINDOW as usize) {
			let prev: Option<u128> = data[i - 1].map(|d| d.timestamp.unique_saturated_into());
			let cur: Option<u128> = data[i].map(|d| d.timestamp.unique_saturated_into());

			let delta = match (prev, cur) {
				(Some(prev), Some(cur)) => cur.saturating_sub(prev),
				_ => block_time.into(),
			};
			ts_delta += delta;
		}

		// sum of all of the differences between timestamps. (betweeen blocks basically)
		if ts_delta == 0 {
			ts_delta = 1;
		}

		let mut diff_sum = U256::zero();
		for i in 0..(DIFFICULTY_ADJUST_WINDOW as usize) {
			let diff = match data[i].map(|d| d.difficulty) {
				Some(diff) => diff,
				None => InitialDifficulty::get(),
			};
			diff_sum += diff; // just summing all of the diff
		}

		if diff_sum < U256::from(MIN_DIFFICULTY) { // minimum...
			diff_sum = U256::from(MIN_DIFFICULTY);
		}

		// adjust time delta toward goal subject to dampening and clamping
		// just the diff algo, not much iportant to understand it now.
		let adj_ts = clamp(
			damp(ts_delta, block_time_window, DIFFICULTY_DAMP_FACTOR),
			block_time_window,
			CLAMP_FACTOR,
		);

		// minimum difficulty avoids getting stuck due to dampening
		let difficulty = min(U256::from(MAX_DIFFICULTY), // just putting the MAX_DIFFICULTY as upperdbound
							 max(U256::from(MIN_DIFFICULTY), // putting MIN_DIFFICULTY as lowerbound
								 diff_sum * U256::from(block_time) / U256::from(adj_ts))); //algo..

		<PastDifficultiesAndTimestamps<T>>::put(data); // saving past diff n timestamps...
		<CurrentDifficulty>::put(difficulty); // saving current difficulty
	}
}
