(define (problem dev_n5_004-P) (:domain depots)
(:objects
	crate0 crate1 crate2 crate3 crate4 - crate
	depot0 - depot
	distributor0 - distributor
	hoist0 hoist1 - hoist
	pallet0 pallet1 pallet2 pallet3 pallet4 - pallet
	truck0 - truck
)
(:init
	(at crate0 depot0)
	(at crate1 depot0)
	(at crate2 distributor0)
	(at crate3 depot0)
	(at crate4 distributor0)
	(at hoist0 depot0)
	(at hoist1 distributor0)
	(at pallet0 depot0)
	(at pallet1 distributor0)
	(at pallet2 distributor0)
	(at pallet3 distributor0)
	(at pallet4 depot0)
	(at truck0 depot0)
	(available hoist0)
	(available hoist1)
	(clear crate3)
	(clear crate4)
	(clear pallet0)
	(clear pallet1)
	(clear pallet2)
	(on crate0 pallet4)
	(on crate1 crate0)
	(on crate2 pallet3)
	(on crate3 crate1)
	(on crate4 crate2)
)
(:goal (and
	(on crate1 pallet2)
	(on crate2 crate1)
	(on crate3 pallet0)
	(on crate4 pallet3)
))
)
