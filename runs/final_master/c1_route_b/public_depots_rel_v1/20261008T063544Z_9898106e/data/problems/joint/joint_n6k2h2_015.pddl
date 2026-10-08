(define (problem depot-1-1-1-6-2-6) (:domain depots)
(:objects
	depot0 - Depot
	distributor0 - Distributor
	truck0 - Truck
	pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 - Pallet
	crate0 crate1 crate2 crate3 crate4 crate5 - Crate
	hoist0 hoist1 - Hoist)
(:init
	(at pallet0 depot0)
	(clear crate1)
	(at pallet1 distributor0)
	(clear pallet1)
	(at pallet2 depot0)
	(clear pallet2)
	(at pallet3 depot0)
	(clear crate2)
	(at pallet4 depot0)
	(clear crate4)
	(at pallet5 depot0)
	(clear crate5)
	(at truck0 distributor0)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at crate0 depot0)
	(on crate0 pallet3)
	(at crate1 depot0)
	(on crate1 pallet0)
	(at crate2 depot0)
	(on crate2 crate0)
	(at crate3 depot0)
	(on crate3 pallet4)
	(at crate4 depot0)
	(on crate4 crate3)
	(at crate5 depot0)
	(on crate5 pallet5)
)

(:goal (and
		(on crate0 pallet0)
		(on crate1 pallet5)
		(on crate2 crate4)
		(on crate3 pallet3)
		(on crate4 pallet2)
		(on crate5 crate1)
	)
))
