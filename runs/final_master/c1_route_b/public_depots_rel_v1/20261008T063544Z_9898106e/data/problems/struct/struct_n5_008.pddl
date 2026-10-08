(define (problem depot-1-1-1-5-2-5) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 - Distributor
	truck0 - Truck
	pallet0 pallet1 pallet2 pallet3 pallet4 - Pallet
	crate0 crate1 crate2 crate3 crate4 - Crate
	hoist0 hoist1 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate4)
	(at pallet1 distributor0)
	(clear crate3)
	(at pallet2 depot0)
	(clear crate2)
	(at pallet3 depot0)
	(clear pallet3)
	(at pallet4 depot0)
	(clear pallet4)
	(at truck0 depot0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at crate0 depot0)
	(on crate0 pallet2)
	(at crate1 distributor0)
	(on crate1 pallet1)
	(at crate2 depot0)
	(on crate2 crate0)
	(at crate3 distributor0)
	(on crate3 crate1)
	(at crate4 depot0)
	(on crate4 pallet0)
)

(:goal (and
		(on crate0 crate2)
		(on crate1 crate3)
		(on crate2 pallet4)
		(on crate3 pallet1)
		(on crate4 crate0)
	)
))
