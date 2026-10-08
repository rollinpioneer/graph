(define (problem depot-1-1-1-4-2-4) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 - Distributor
	truck0 - Truck
	pallet0 pallet1 pallet2 pallet3 - Pallet
	crate0 crate1 crate2 crate3 - Crate
	hoist0 hoist1 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate1)
	(at pallet1 distributor0)
	(clear pallet1)
	(at pallet2 depot0)
	(clear pallet2)
	(at pallet3 depot0)
	(clear crate3)
	(at truck0 distributor0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at crate0 depot0)
	(on crate0 pallet0)
	(at crate1 depot0)
	(on crate1 crate0)
	(at crate2 depot0)
	(on crate2 pallet3)
	(at crate3 depot0)
	(on crate3 crate2)
)

(:goal (and
		(on crate0 crate3)
		(on crate1 pallet2)
		(on crate2 crate1)
		(on crate3 pallet0)
	)
))
