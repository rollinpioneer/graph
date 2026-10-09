(define (problem dev_n4_005-P) (:domain depots)
(:objects
	crate0 crate1 crate2 crate3 - crate
	depot0 - depot
	distributor0 - distributor
	hoist0 hoist1 - hoist
	pallet0 pallet1 pallet2 pallet3 - pallet
	truck0 - truck
)
(:init
	(at crate0 distributor0)
	(at crate1 distributor0)
	(at crate2 distributor0)
	(at crate3 depot0)
	(at hoist0 depot0)
	(at hoist1 distributor0)
	(at pallet0 depot0)
	(at pallet1 distributor0)
	(at pallet2 depot0)
	(at pallet3 distributor0)
	(at truck0 distributor0)
	(available hoist0)
	(available hoist1)
	(clear crate1)
	(clear crate2)
	(clear crate3)
	(clear pallet2)
	(on crate0 pallet3)
	(on crate1 crate0)
	(on crate2 pallet1)
	(on crate3 pallet0)
)
(:goal (and
	(on crate1 crate2)
	(on crate2 pallet3)
	(on crate3 pallet2)
))
)
