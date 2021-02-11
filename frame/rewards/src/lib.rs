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

//! Reward handling module for Kulupu.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)] // needed only for test
mod mock;
#[cfg(test)] // needed only for test
mod tests;
#[cfg(feature = "runtime-benchmarks")] // only of runtime-benchmark
mod benchmarking; 
mod default_weights;
mod migrations;

// like includes
use codec::{Encode, Decode};
use sp_std::{result, ops::Bound::Included, prelude::*, collections::btree_map::BTreeMap};
use sp_runtime::{RuntimeDebug, Perbill, traits::{Saturating, Zero}};
use sp_inherents::{InherentIdentifier, InherentData, ProvideInherent, IsFatalError};
use sp_consensus_pow::POW_ENGINE_ID;
#[cfg(feature = "std")]
use sp_inherents::ProvideInherentData;
use frame_support::{
	decl_module, decl_storage, decl_error, decl_event, ensure,
	traits::{Get, Currency, LockIdentifier, LockableCurrency, WithdrawReasons},
	weights::Weight,
};
use frame_system::{ensure_root, ensure_signed};

/// Trait for generating reward locks.
pub trait GenerateRewardLocks<T: Config> {
	/// Generate reward locks.
	fn generate_reward_locks( // like declaring the function lol
// types of the two arguments to calculate the reward lock of course
		current_block: T::BlockNumber,
		total_reward: BalanceOf<T>,
// return type
	) -> BTreeMap<T::BlockNumber, BalanceOf<T>>;

	fn max_locks() -> u32;
}

// implementation...
impl<T: Config> GenerateRewardLocks<T> for () {
	fn generate_reward_locks(
		_current_block: T::BlockNumber, // _ not warning for unused
		_total_reward: BalanceOf<T>,
	) -> BTreeMap<T::BlockNumber, BalanceOf<T>> { // like an hashmap but with more functions
		Default::default() // where is it...?
	}

	fn max_locks() -> u32 {
		0
	}
}

pub trait WeightInfo {
	fn on_initialize() -> Weight; // this type has been inlcuded with `use` before.
	fn on_finalize() -> Weight;
	fn note_author_prefs() -> Weight;
	fn set_reward() -> Weight;
	fn set_taxation() -> Weight;
	fn unlock() -> Weight;
	fn set_curve(_l: u32, ) -> Weight;
	fn fund() -> Weight;
	fn set_additional_rewards() -> Weight;
}

/// Config for rewards.
pub trait Config: frame_system::Config {
	/// The overarching event type.
	type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
	/// An implementation of on-chain currency.
	type Currency: LockableCurrency<Self::AccountId>;
	/// Donation destination.
	type DonationDestination: Get<Self::AccountId>;
	/// Generate reward locks.
	type GenerateRewardLocks: GenerateRewardLocks<Self>;
	/// Weights for this pallet.
	type WeightInfo: WeightInfo;
}

/// Type alias for currency balance.
pub type BalanceOf<T> = <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

decl_error! {
	pub enum Error for Module<T: Config> {
		/// Reward set is too low.
		RewardTooLow,
		/// Mint value is too low.
		MintTooLow,
		/// Reward curve is not sorted.
		NotSorted,
	}
}

// seems easy..
decl_storage! {
	trait Store for Module<T: Config> as Rewards {
		/// Current block author.
		Author get(fn author): Option<T::AccountId>; //TODO not sure about it, probably ill get it later when used

		/// Current block reward for miner.
		Reward get(fn reward) config(): BalanceOf<T>; // reward
		/// Pending reward locks.   // get function aliased as reward_lock is seems
		RewardLocks get(fn reward_locks): map hasher(twox_64_concat) T::AccountId => BTreeMap<T::BlockNumber, BalanceOf<T>>;
		/// Reward changes planned in the future.
		RewardChanges get(fn reward_changes): BTreeMap<T::BlockNumber, BalanceOf<T>>;

		/// Current block mints.
		Mints get(fn mints) config(): BTreeMap<T::AccountId, BalanceOf<T>>; // donation
		/// Mint changes planned in the future.
		MintChanges get(fn mint_changes): BTreeMap<T::BlockNumber, BTreeMap<T::AccountId, BalanceOf<T>>>;

		StorageVersion build(|_| migrations::StorageVersion::V1): migrations::StorageVersion;
	}
}

decl_event! {
	pub enum Event<T> where AccountId = <T as frame_system::Config>::AccountId, Balance = BalanceOf<T> {
		/// A new schedule has been set.
		ScheduleSet, // TODO ill get this later i guess
		/// Reward has been sent.
		Rewarded(AccountId, Balance),
		/// Reward has been changed.
		RewardChanged(Balance),
		/// Mint has been sent.
		Minted(AccountId, Balance),
		/// Mint has been changed.
		MintsChanged(BTreeMap<AccountId, Balance>),
	}
}

decl_module! {
	pub struct Module<T: Config> for enum Call where origin: T::Origin { // implementing the struct THAT'S the module itself.
		type Error = Error<T>; // ok

		fn deposit_event() = default; // OK

		fn on_initialize(now: T::BlockNumber) -> Weight { // ON_ means happen on events..
// To initialize a block, the System module and all other included runtime modules have their on_initialize 
// function called which executes any business logic defined by those modules to take place before transactions are 
// executed. The modules are executed in the order which they are defined in the construct_runtime! macro, but with the 
// System module always executing first.
			let author = frame_system::Module::<T>::digest()  	// System digest item that contains the root of changes trie at given block. 
										// It is created for 
										// every block iff runtime supports changes trie creation.
				.logs // Get reference to all digest items. pub fn logs(&self) -> &[DigestItem<Hash>]
				.iter() // returns the iterator
// A pre-runtime digest.
// These are messages from the consensus engine to the runtime, although the consensus engine can (and should) read them 
// itself to avoid code and state duplication. It is erroneous for a runtime to produce these, but this is not (yet) checked.
// NOTE: the runtime is not allowed to panic or fail in an on_initialize call if an expected PreRuntime digest is not present. 
// It is the responsibility of a external block verifier to check this. Runtime API calls will initialize the block without pre-runtime digests, 
// so initialization cannot fail when they are missing.

				.filter_map(|s| s.as_pre_runtime()) // filtering i guess https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.filter_map
				.filter_map(|(id, mut data)| if id == POW_ENGINE_ID { // if the pow engine ID matches (i guess like version..? for hard forks)
					// i guess decoding digest to the account id... ok() converst result to option
					T::AccountId::decode(&mut data).ok() //it's kinda doing it for everyone instead of doing it just for the one we need...
				} else {
					None
				})
				.next(); // returns none when iteration is finished.

			if let Some(author) = author { // if author isn't None basically
				<Self as Store>::Author::put(author); // putting the author...
			}

			//https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html
			// It`s like set get etc, its a function of storage of substrate it seems.
			RewardChanges::<T>::mutate(|reward_changes| {
				let mut removing = Vec::new();
				
				// searches occurrences in that range, each occurence is a reward change..
				for (block_number, reward) in reward_changes.range((Included(Zero::zero()), Included(now))) { // now is the block number...
					Reward::<T>::set(*reward); // *reward.. so the value!!!
								   // reward has been set in decl_starage...
								   // couldnt it fail ?!?
					removing.push(*block_number); we have to remove this block number... dereferenced

					Self::deposit_event(Event::<T>::RewardChanged(*reward)); // emitting the event *event so the value!!!
				}

				for block_number in removing { // removing the block numbers for which we ve already did the reward change
					reward_changes.remove(&block_number); // to remove it.. it must be passed the reference, dunno why anyway.
				}
			});
		
			// same as above
			MintChanges::<T>::mutate(|mint_changes| {
				let mut removing = Vec::new();

				for (block_number, mints) in mint_changes.range((Included(Zero::zero()), Included(now))) {
					Mints::<T>::set(mints.clone());
					removing.push(*block_number);

					Self::deposit_event(Event::<T>::MintsChanged(mints.clone()));
				}

				for block_number in removing {
					mint_changes.remove(&block_number);
				}
			});

			T::WeightInfo::on_initialize().saturating_add(T::WeightInfo::on_finalize())
		}

// After all queued extrinsics have been executed, the Executive module calls into each module's on_finalize function to perform any final
// business logic which should take place at the end of the block. The modules are again executed in the order which they are defined in the 
// construct_runtime! macro, but in this case, the System module finalizes last.

		fn on_finalize(now: T::BlockNumber) {
			if let Some(author) = <Self as Store>::Author::get() { // if it`s not None
				let reward = Reward::<T>::get(); // reward amount
				Self::do_reward(&author, reward, now); // function to pay
			}

			let mints = Mints::<T>::get(); //mint amount
			Self::do_mints(&mints); //pay mint

			<Self as Store>::Author::kill(); // deletes it
		}

		fn on_runtime_upgrade() -> frame_support::weights::Weight {
			let version = StorageVersion::get();
			let new_version = version.migrate::<T>();
			StorageVersion::put(new_version);

			0 //why?
		}

		#[weight = 0] // probably because its done by the root anyway...
		fn set_schedule(
			origin,
			reward: BalanceOf<T>,
			mints: BTreeMap<T::AccountId, BalanceOf<T>>,
			reward_changes: BTreeMap<T::BlockNumber, BalanceOf<T>>,
			mint_changes: BTreeMap<T::BlockNumber, BTreeMap<T::AccountId, BalanceOf<T>>>,
		) {
			ensure_root(origin)?; // must be the root
			
			// checking errors...
			ensure!(reward >= T::Currency::minimum_balance(), Error::<T>::RewardTooLow);
			for (_, mint) in &mints {
				ensure!(*mint >= T::Currency::minimum_balance(), Error::<T>::MintTooLow);
			}
			for (_, reward_change) in &reward_changes {
				ensure!(*reward_change >= T::Currency::minimum_balance(), Error::<T>::RewardTooLow);
			}
			for (_, mint_change) in &mint_changes {
				for (_, mint) in mint_change {
					ensure!(*mint >= T::Currency::minimum_balance(), Error::<T>::MintTooLow);
				}
			}

			Reward::<T>::put(reward);
			Self::deposit_event(RawEvent::RewardChanged(reward));

			Mints::<T>::put(mints.clone());
			Self::deposit_event(RawEvent::MintsChanged(mints));

			RewardChanges::<T>::put(reward_changes);
			MintChanges::<T>::put(mint_changes);
			Self::deposit_event(RawEvent::ScheduleSet);
		}

		/// Unlock any vested rewards for `target` account.
		#[weight = T::WeightInfo::unlock()]
		fn unlock(origin, target: T::AccountId) {
			ensure_signed(origin)?;

			let locks = Self::reward_locks(&target);
			let current_number = frame_system::Module::<T>::block_number(); // current block
			Self::do_update_reward_locks(&target, locks, current_number); // actual function..
		}
	}
}

const REWARDS_ID: LockIdentifier = *b"rewards "; // TODO not sure... seems just an ascii string

// help functions
impl<T: Config> Module<T> {
	fn do_reward(author: &T::AccountId, reward: BalanceOf<T>, when: T::BlockNumber) {
		let miner_total = reward;

		let miner_reward_locks = T::GenerateRewardLocks::generate_reward_locks( // calling the function
			when,
			miner_total,
		);

		// drop is destructor
		drop(T::Currency::deposit_creating(&author, miner_total)); // depositing the coins

		if miner_reward_locks.len() > 0 { //why shouldn`t it be ?
			let mut locks = Self::reward_locks(&author); // get the other locks

			for (new_lock_number, new_lock_balance) in miner_reward_locks { // for each new unlock that will happen
				// Returns the contained Some value or a provided default... 
				// TODO &BalanceOf::<T>::default() what`s this default ?
				let old_balance = *locks.get(&new_lock_number).unwrap_or(&BalanceOf::<T>::default()); // get old balance
				// saturating doesn`t overflow (it means it will just saturate...)
				let new_balance = old_balance.saturating_add(new_lock_balance); // create new balance
				locks.insert(new_lock_number, new_balance);
			}

			Self::do_update_reward_locks(&author, locks, when); // unlock what`s unlockable
		}
	}

	fn do_update_reward_locks(
		author: &T::AccountId,
		mut locks: BTreeMap<T::BlockNumber, BalanceOf<T>>,
		current_number: T::BlockNumber
	) {
		let mut expired = Vec::new();
		let mut total_locked: BalanceOf<T> = Zero::zero();

		for (block_number, locked_balance) in &locks {
			if block_number <= &current_number {
				expired.push(*block_number);
			} else {
				total_locked = total_locked.saturating_add(*locked_balance);
			}
		}

		for block_number in expired {
			locks.remove(&block_number);
		}

		T::Currency::set_lock(
			REWARDS_ID,
			&author,
			total_locked, // new total lock
			WithdrawReasons::except(WithdrawReasons::TRANSACTION_PAYMENT),
		);

		<Self as Store>::RewardLocks::insert(author, locks);
	}

	fn do_mints(
		mints: &BTreeMap<T::AccountId, BalanceOf<T>>,
	) {
		for (destination, mint) in mints {
			drop(T::Currency::deposit_creating(&destination, *mint)); // depositing
		}
	}
}

pub const INHERENT_IDENTIFIER_V0: InherentIdentifier = *b"rewards_";
pub const INHERENT_IDENTIFIER: InherentIdentifier = *b"rewards1";

#[derive(Encode, Decode, RuntimeDebug)]
pub enum InherentError { }

impl IsFatalError for InherentError {
	fn is_fatal_error(&self) -> bool {
		match *self { }
	}
}

impl InherentError {
	/// Try to create an instance ouf of the given identifier and data.
	#[cfg(feature = "std")]
	pub fn try_from(id: &InherentIdentifier, data: &[u8]) -> Option<Self> {
		if id == &INHERENT_IDENTIFIER {
			<InherentError as codec::Decode>::decode(&mut &data[..]).ok()
		} else {
			None
		}
	}
}

#[cfg(feature = "std")]
pub struct InherentDataProviderV0(pub Vec<u8>);

#[cfg(feature = "std")]
impl ProvideInherentData for InherentDataProviderV0 {
	fn inherent_identifier(&self) -> &'static InherentIdentifier {
		&INHERENT_IDENTIFIER_V0
	}

	fn provide_inherent_data(
		&self,
		inherent_data: &mut InherentData
	) -> Result<(), sp_inherents::Error> {
		inherent_data.put_data(INHERENT_IDENTIFIER_V0, &self.0)
	}

	fn error_to_string(&self, error: &[u8]) -> Option<String> {
		InherentError::try_from(&INHERENT_IDENTIFIER_V0, error).map(|e| format!("{:?}", e))
	}
}

pub type InherentType = (Vec<u8>, Perbill);

#[cfg(feature = "std")]
pub struct InherentDataProvider(pub InherentType);

#[cfg(feature = "std")]
impl ProvideInherentData for InherentDataProvider {
	fn inherent_identifier(&self) -> &'static InherentIdentifier {
		&INHERENT_IDENTIFIER
	}

	fn provide_inherent_data(
		&self,
		inherent_data: &mut InherentData
	) -> Result<(), sp_inherents::Error> {
		inherent_data.put_data(INHERENT_IDENTIFIER, &self.0)
	}

	fn error_to_string(&self, error: &[u8]) -> Option<String> {
		InherentError::try_from(&INHERENT_IDENTIFIER, error).map(|e| format!("{:?}", e))
	}
}

impl<T: Config> ProvideInherent for Module<T> {
	type Call = Call<T>;
	type Error = InherentError;
	const INHERENT_IDENTIFIER: InherentIdentifier = INHERENT_IDENTIFIER;

	fn create_inherent(_data: &InherentData) -> Option<Self::Call> {
		None
	}

	fn check_inherent(_call: &Self::Call, _data: &InherentData) -> result::Result<(), Self::Error> {
		Ok(())
	}
}
