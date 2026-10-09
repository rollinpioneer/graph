(define (problem dev_n5_003-W) (:domain depots)
(:objects
	crate0 crate1 crate2 crate3 crate4 - crate
	depot0 - depot
	distributor0 distributor1 - distributor
	hoist0 hoist1 hoist2 - hoist
	pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 - pallet
	truck0 - truck
)
(:init
	(at crate0 depot0)
	(at crate1 distributor1)
	(at crate2 depot0)
	(at crate3 depot0)
	(at crate4 depot0)
	(at hoist0 depot0)
	(at hoist1 distributor0)
	(at pallet0 distributor1)
	(at pallet1 distributor0)
	(at pallet2 depot0)
	(at pallet3 depot0)
	(at pallet4 depot0)
	(at truck0 depot0)
	(available hoist0)
	(available hoist1)
	(clear crate0)
	(clear crate1)
	(clear crate3)
	(clear crate4)
	(clear pallet1)
	(on crate0 pallet2)
	(on crate1 pallet0)
	(on crate2 pallet4)
	(on crate3 crate2)
	(on crate4 pallet3)
	(at hoist2 distributor1)
	(available hoist2)
	(at pallet5 distributor1)
	(clear pallet5)
)
(:goal (and
	(on crate0 pallet4)
	(on crate1 crate0)
	(on crate2 pallet3)
	(on crate3 pallet1)
	(on crate4 pallet2)
))
)
