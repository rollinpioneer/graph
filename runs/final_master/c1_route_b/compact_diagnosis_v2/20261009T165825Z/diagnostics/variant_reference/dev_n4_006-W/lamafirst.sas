begin_version
3
end_version
begin_metric
0
end_metric
21
begin_variable
var0
-1
3
Atom at(truck0, depot0)
Atom at(truck0, distributor0)
Atom at(truck0, distributor1)
end_variable
begin_variable
var1
-1
2
Atom clear(pallet0)
NegatedAtom clear(pallet0)
end_variable
begin_variable
var2
-1
2
Atom clear(pallet1)
NegatedAtom clear(pallet1)
end_variable
begin_variable
var3
-1
2
Atom clear(pallet2)
NegatedAtom clear(pallet2)
end_variable
begin_variable
var4
-1
2
Atom clear(pallet3)
NegatedAtom clear(pallet3)
end_variable
begin_variable
var5
-1
2
Atom clear(pallet4)
NegatedAtom clear(pallet4)
end_variable
begin_variable
var6
-1
4
Atom at(crate0, depot0)
Atom at(crate0, distributor0)
Atom at(crate0, distributor1)
<none of those>
end_variable
begin_variable
var7
-1
4
Atom at(crate1, depot0)
Atom at(crate1, distributor0)
Atom at(crate1, distributor1)
<none of those>
end_variable
begin_variable
var8
-1
4
Atom at(crate2, depot0)
Atom at(crate2, distributor0)
Atom at(crate2, distributor1)
<none of those>
end_variable
begin_variable
var9
-1
4
Atom at(crate3, depot0)
Atom at(crate3, distributor0)
Atom at(crate3, distributor1)
<none of those>
end_variable
begin_variable
var10
-1
2
Atom available(hoist0)
NegatedAtom available(hoist0)
end_variable
begin_variable
var11
-1
2
Atom clear(crate0)
NegatedAtom clear(crate0)
end_variable
begin_variable
var12
-1
2
Atom clear(crate1)
NegatedAtom clear(crate1)
end_variable
begin_variable
var13
-1
2
Atom available(hoist1)
NegatedAtom available(hoist1)
end_variable
begin_variable
var14
-1
2
Atom clear(crate2)
NegatedAtom clear(crate2)
end_variable
begin_variable
var15
-1
2
Atom clear(crate3)
NegatedAtom clear(crate3)
end_variable
begin_variable
var16
-1
2
Atom available(hoist2)
NegatedAtom available(hoist2)
end_variable
begin_variable
var17
-1
12
Atom in(crate0, truck0)
Atom lifting(hoist0, crate0)
Atom lifting(hoist1, crate0)
Atom lifting(hoist2, crate0)
Atom on(crate0, crate1)
Atom on(crate0, crate2)
Atom on(crate0, crate3)
Atom on(crate0, pallet0)
Atom on(crate0, pallet1)
Atom on(crate0, pallet2)
Atom on(crate0, pallet3)
Atom on(crate0, pallet4)
end_variable
begin_variable
var18
-1
12
Atom in(crate1, truck0)
Atom lifting(hoist0, crate1)
Atom lifting(hoist1, crate1)
Atom lifting(hoist2, crate1)
Atom on(crate1, crate0)
Atom on(crate1, crate2)
Atom on(crate1, crate3)
Atom on(crate1, pallet0)
Atom on(crate1, pallet1)
Atom on(crate1, pallet2)
Atom on(crate1, pallet3)
Atom on(crate1, pallet4)
end_variable
begin_variable
var19
-1
12
Atom in(crate2, truck0)
Atom lifting(hoist0, crate2)
Atom lifting(hoist1, crate2)
Atom lifting(hoist2, crate2)
Atom on(crate2, crate0)
Atom on(crate2, crate1)
Atom on(crate2, crate3)
Atom on(crate2, pallet0)
Atom on(crate2, pallet1)
Atom on(crate2, pallet2)
Atom on(crate2, pallet3)
Atom on(crate2, pallet4)
end_variable
begin_variable
var20
-1
12
Atom in(crate3, truck0)
Atom lifting(hoist0, crate3)
Atom lifting(hoist1, crate3)
Atom lifting(hoist2, crate3)
Atom on(crate3, crate0)
Atom on(crate3, crate1)
Atom on(crate3, crate2)
Atom on(crate3, pallet0)
Atom on(crate3, pallet1)
Atom on(crate3, pallet2)
Atom on(crate3, pallet3)
Atom on(crate3, pallet4)
end_variable
16
begin_mutex_group
7
6 0
6 1
6 2
17 0
17 1
17 2
17 3
end_mutex_group
begin_mutex_group
7
7 0
7 1
7 2
18 0
18 1
18 2
18 3
end_mutex_group
begin_mutex_group
7
8 0
8 1
8 2
19 0
19 1
19 2
19 3
end_mutex_group
begin_mutex_group
7
9 0
9 1
9 2
20 0
20 1
20 2
20 3
end_mutex_group
begin_mutex_group
5
10 0
17 1
18 1
19 1
20 1
end_mutex_group
begin_mutex_group
5
13 0
17 2
18 2
19 2
20 2
end_mutex_group
begin_mutex_group
5
16 0
17 3
18 3
19 3
20 3
end_mutex_group
begin_mutex_group
8
11 0
17 0
17 1
17 2
17 3
18 4
19 4
20 4
end_mutex_group
begin_mutex_group
8
12 0
17 4
18 0
18 1
18 2
18 3
19 5
20 5
end_mutex_group
begin_mutex_group
8
14 0
17 5
18 5
19 0
19 1
19 2
19 3
20 6
end_mutex_group
begin_mutex_group
8
15 0
17 6
18 6
19 6
20 0
20 1
20 2
20 3
end_mutex_group
begin_mutex_group
5
1 0
17 7
18 7
19 7
20 7
end_mutex_group
begin_mutex_group
5
2 0
17 8
18 8
19 8
20 8
end_mutex_group
begin_mutex_group
5
3 0
17 9
18 9
19 9
20 9
end_mutex_group
begin_mutex_group
5
4 0
17 10
18 10
19 10
20 10
end_mutex_group
begin_mutex_group
5
5 0
17 11
18 11
19 11
20 11
end_mutex_group
begin_state
1
0
1
0
0
0
2
2
2
2
0
1
1
0
1
0
0
8
4
5
6
end_state
begin_goal
4
17 5
18 10
19 9
20 8
end_goal
182
begin_operator
drive truck0 depot0 distributor0
0
1
0 0 0 1
1
end_operator
begin_operator
drive truck0 depot0 distributor1
0
1
0 0 0 2
1
end_operator
begin_operator
drive truck0 distributor0 depot0
0
1
0 0 1 0
1
end_operator
begin_operator
drive truck0 distributor0 distributor1
0
1
0 0 1 2
1
end_operator
begin_operator
drive truck0 distributor1 depot0
0
1
0 0 2 0
1
end_operator
begin_operator
drive truck0 distributor1 distributor0
0
1
0 0 2 1
1
end_operator
begin_operator
drop hoist0 crate0 crate1 depot0
1
7 0
5
0 6 -1 0
0 10 -1 0
0 11 -1 0
0 12 0 1
0 17 1 4
1
end_operator
begin_operator
drop hoist0 crate0 crate2 depot0
1
8 0
5
0 6 -1 0
0 10 -1 0
0 11 -1 0
0 14 0 1
0 17 1 5
1
end_operator
begin_operator
drop hoist0 crate0 crate3 depot0
1
9 0
5
0 6 -1 0
0 10 -1 0
0 11 -1 0
0 15 0 1
0 17 1 6
1
end_operator
begin_operator
drop hoist0 crate0 pallet0 depot0
0
5
0 6 -1 0
0 10 -1 0
0 11 -1 0
0 1 0 1
0 17 1 7
1
end_operator
begin_operator
drop hoist0 crate1 crate0 depot0
1
6 0
5
0 7 -1 0
0 10 -1 0
0 11 0 1
0 12 -1 0
0 18 1 4
1
end_operator
begin_operator
drop hoist0 crate1 crate2 depot0
1
8 0
5
0 7 -1 0
0 10 -1 0
0 12 -1 0
0 14 0 1
0 18 1 5
1
end_operator
begin_operator
drop hoist0 crate1 crate3 depot0
1
9 0
5
0 7 -1 0
0 10 -1 0
0 12 -1 0
0 15 0 1
0 18 1 6
1
end_operator
begin_operator
drop hoist0 crate1 pallet0 depot0
0
5
0 7 -1 0
0 10 -1 0
0 12 -1 0
0 1 0 1
0 18 1 7
1
end_operator
begin_operator
drop hoist0 crate2 crate0 depot0
1
6 0
5
0 8 -1 0
0 10 -1 0
0 11 0 1
0 14 -1 0
0 19 1 4
1
end_operator
begin_operator
drop hoist0 crate2 crate1 depot0
1
7 0
5
0 8 -1 0
0 10 -1 0
0 12 0 1
0 14 -1 0
0 19 1 5
1
end_operator
begin_operator
drop hoist0 crate2 crate3 depot0
1
9 0
5
0 8 -1 0
0 10 -1 0
0 14 -1 0
0 15 0 1
0 19 1 6
1
end_operator
begin_operator
drop hoist0 crate2 pallet0 depot0
0
5
0 8 -1 0
0 10 -1 0
0 14 -1 0
0 1 0 1
0 19 1 7
1
end_operator
begin_operator
drop hoist0 crate3 crate0 depot0
1
6 0
5
0 9 -1 0
0 10 -1 0
0 11 0 1
0 15 -1 0
0 20 1 4
1
end_operator
begin_operator
drop hoist0 crate3 crate1 depot0
1
7 0
5
0 9 -1 0
0 10 -1 0
0 12 0 1
0 15 -1 0
0 20 1 5
1
end_operator
begin_operator
drop hoist0 crate3 crate2 depot0
1
8 0
5
0 9 -1 0
0 10 -1 0
0 14 0 1
0 15 -1 0
0 20 1 6
1
end_operator
begin_operator
drop hoist0 crate3 pallet0 depot0
0
5
0 9 -1 0
0 10 -1 0
0 15 -1 0
0 1 0 1
0 20 1 7
1
end_operator
begin_operator
drop hoist1 crate0 crate1 distributor0
1
7 1
5
0 6 -1 1
0 13 -1 0
0 11 -1 0
0 12 0 1
0 17 2 4
1
end_operator
begin_operator
drop hoist1 crate0 crate2 distributor0
1
8 1
5
0 6 -1 1
0 13 -1 0
0 11 -1 0
0 14 0 1
0 17 2 5
1
end_operator
begin_operator
drop hoist1 crate0 crate3 distributor0
1
9 1
5
0 6 -1 1
0 13 -1 0
0 11 -1 0
0 15 0 1
0 17 2 6
1
end_operator
begin_operator
drop hoist1 crate0 pallet2 distributor0
0
5
0 6 -1 1
0 13 -1 0
0 11 -1 0
0 3 0 1
0 17 2 9
1
end_operator
begin_operator
drop hoist1 crate0 pallet3 distributor0
0
5
0 6 -1 1
0 13 -1 0
0 11 -1 0
0 4 0 1
0 17 2 10
1
end_operator
begin_operator
drop hoist1 crate1 crate0 distributor0
1
6 1
5
0 7 -1 1
0 13 -1 0
0 11 0 1
0 12 -1 0
0 18 2 4
1
end_operator
begin_operator
drop hoist1 crate1 crate2 distributor0
1
8 1
5
0 7 -1 1
0 13 -1 0
0 12 -1 0
0 14 0 1
0 18 2 5
1
end_operator
begin_operator
drop hoist1 crate1 crate3 distributor0
1
9 1
5
0 7 -1 1
0 13 -1 0
0 12 -1 0
0 15 0 1
0 18 2 6
1
end_operator
begin_operator
drop hoist1 crate1 pallet2 distributor0
0
5
0 7 -1 1
0 13 -1 0
0 12 -1 0
0 3 0 1
0 18 2 9
1
end_operator
begin_operator
drop hoist1 crate1 pallet3 distributor0
0
5
0 7 -1 1
0 13 -1 0
0 12 -1 0
0 4 0 1
0 18 2 10
1
end_operator
begin_operator
drop hoist1 crate2 crate0 distributor0
1
6 1
5
0 8 -1 1
0 13 -1 0
0 11 0 1
0 14 -1 0
0 19 2 4
1
end_operator
begin_operator
drop hoist1 crate2 crate1 distributor0
1
7 1
5
0 8 -1 1
0 13 -1 0
0 12 0 1
0 14 -1 0
0 19 2 5
1
end_operator
begin_operator
drop hoist1 crate2 crate3 distributor0
1
9 1
5
0 8 -1 1
0 13 -1 0
0 14 -1 0
0 15 0 1
0 19 2 6
1
end_operator
begin_operator
drop hoist1 crate2 pallet2 distributor0
0
5
0 8 -1 1
0 13 -1 0
0 14 -1 0
0 3 0 1
0 19 2 9
1
end_operator
begin_operator
drop hoist1 crate2 pallet3 distributor0
0
5
0 8 -1 1
0 13 -1 0
0 14 -1 0
0 4 0 1
0 19 2 10
1
end_operator
begin_operator
drop hoist1 crate3 crate0 distributor0
1
6 1
5
0 9 -1 1
0 13 -1 0
0 11 0 1
0 15 -1 0
0 20 2 4
1
end_operator
begin_operator
drop hoist1 crate3 crate1 distributor0
1
7 1
5
0 9 -1 1
0 13 -1 0
0 12 0 1
0 15 -1 0
0 20 2 5
1
end_operator
begin_operator
drop hoist1 crate3 crate2 distributor0
1
8 1
5
0 9 -1 1
0 13 -1 0
0 14 0 1
0 15 -1 0
0 20 2 6
1
end_operator
begin_operator
drop hoist1 crate3 pallet2 distributor0
0
5
0 9 -1 1
0 13 -1 0
0 15 -1 0
0 3 0 1
0 20 2 9
1
end_operator
begin_operator
drop hoist1 crate3 pallet3 distributor0
0
5
0 9 -1 1
0 13 -1 0
0 15 -1 0
0 4 0 1
0 20 2 10
1
end_operator
begin_operator
drop hoist2 crate0 crate1 distributor1
1
7 2
5
0 6 -1 2
0 16 -1 0
0 11 -1 0
0 12 0 1
0 17 3 4
1
end_operator
begin_operator
drop hoist2 crate0 crate2 distributor1
1
8 2
5
0 6 -1 2
0 16 -1 0
0 11 -1 0
0 14 0 1
0 17 3 5
1
end_operator
begin_operator
drop hoist2 crate0 crate3 distributor1
1
9 2
5
0 6 -1 2
0 16 -1 0
0 11 -1 0
0 15 0 1
0 17 3 6
1
end_operator
begin_operator
drop hoist2 crate0 pallet1 distributor1
0
5
0 6 -1 2
0 16 -1 0
0 11 -1 0
0 2 0 1
0 17 3 8
1
end_operator
begin_operator
drop hoist2 crate0 pallet4 distributor1
0
5
0 6 -1 2
0 16 -1 0
0 11 -1 0
0 5 0 1
0 17 3 11
1
end_operator
begin_operator
drop hoist2 crate1 crate0 distributor1
1
6 2
5
0 7 -1 2
0 16 -1 0
0 11 0 1
0 12 -1 0
0 18 3 4
1
end_operator
begin_operator
drop hoist2 crate1 crate2 distributor1
1
8 2
5
0 7 -1 2
0 16 -1 0
0 12 -1 0
0 14 0 1
0 18 3 5
1
end_operator
begin_operator
drop hoist2 crate1 crate3 distributor1
1
9 2
5
0 7 -1 2
0 16 -1 0
0 12 -1 0
0 15 0 1
0 18 3 6
1
end_operator
begin_operator
drop hoist2 crate1 pallet1 distributor1
0
5
0 7 -1 2
0 16 -1 0
0 12 -1 0
0 2 0 1
0 18 3 8
1
end_operator
begin_operator
drop hoist2 crate1 pallet4 distributor1
0
5
0 7 -1 2
0 16 -1 0
0 12 -1 0
0 5 0 1
0 18 3 11
1
end_operator
begin_operator
drop hoist2 crate2 crate0 distributor1
1
6 2
5
0 8 -1 2
0 16 -1 0
0 11 0 1
0 14 -1 0
0 19 3 4
1
end_operator
begin_operator
drop hoist2 crate2 crate1 distributor1
1
7 2
5
0 8 -1 2
0 16 -1 0
0 12 0 1
0 14 -1 0
0 19 3 5
1
end_operator
begin_operator
drop hoist2 crate2 crate3 distributor1
1
9 2
5
0 8 -1 2
0 16 -1 0
0 14 -1 0
0 15 0 1
0 19 3 6
1
end_operator
begin_operator
drop hoist2 crate2 pallet1 distributor1
0
5
0 8 -1 2
0 16 -1 0
0 14 -1 0
0 2 0 1
0 19 3 8
1
end_operator
begin_operator
drop hoist2 crate2 pallet4 distributor1
0
5
0 8 -1 2
0 16 -1 0
0 14 -1 0
0 5 0 1
0 19 3 11
1
end_operator
begin_operator
drop hoist2 crate3 crate0 distributor1
1
6 2
5
0 9 -1 2
0 16 -1 0
0 11 0 1
0 15 -1 0
0 20 3 4
1
end_operator
begin_operator
drop hoist2 crate3 crate1 distributor1
1
7 2
5
0 9 -1 2
0 16 -1 0
0 12 0 1
0 15 -1 0
0 20 3 5
1
end_operator
begin_operator
drop hoist2 crate3 crate2 distributor1
1
8 2
5
0 9 -1 2
0 16 -1 0
0 14 0 1
0 15 -1 0
0 20 3 6
1
end_operator
begin_operator
drop hoist2 crate3 pallet1 distributor1
0
5
0 9 -1 2
0 16 -1 0
0 15 -1 0
0 2 0 1
0 20 3 8
1
end_operator
begin_operator
drop hoist2 crate3 pallet4 distributor1
0
5
0 9 -1 2
0 16 -1 0
0 15 -1 0
0 5 0 1
0 20 3 11
1
end_operator
begin_operator
lift hoist0 crate0 crate1 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 12 -1 0
0 17 4 1
1
end_operator
begin_operator
lift hoist0 crate0 crate2 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 14 -1 0
0 17 5 1
1
end_operator
begin_operator
lift hoist0 crate0 crate3 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 15 -1 0
0 17 6 1
1
end_operator
begin_operator
lift hoist0 crate0 pallet0 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 1 -1 0
0 17 7 1
1
end_operator
begin_operator
lift hoist0 crate0 pallet1 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 2 -1 0
0 17 8 1
1
end_operator
begin_operator
lift hoist0 crate0 pallet2 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 3 -1 0
0 17 9 1
1
end_operator
begin_operator
lift hoist0 crate0 pallet3 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 4 -1 0
0 17 10 1
1
end_operator
begin_operator
lift hoist0 crate0 pallet4 depot0
0
5
0 6 0 3
0 10 0 1
0 11 0 1
0 5 -1 0
0 17 11 1
1
end_operator
begin_operator
lift hoist0 crate1 crate0 depot0
0
5
0 7 0 3
0 10 0 1
0 11 -1 0
0 12 0 1
0 18 4 1
1
end_operator
begin_operator
lift hoist0 crate1 crate2 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 14 -1 0
0 18 5 1
1
end_operator
begin_operator
lift hoist0 crate1 crate3 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 15 -1 0
0 18 6 1
1
end_operator
begin_operator
lift hoist0 crate1 pallet0 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 1 -1 0
0 18 7 1
1
end_operator
begin_operator
lift hoist0 crate1 pallet1 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 2 -1 0
0 18 8 1
1
end_operator
begin_operator
lift hoist0 crate1 pallet2 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 3 -1 0
0 18 9 1
1
end_operator
begin_operator
lift hoist0 crate1 pallet3 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 4 -1 0
0 18 10 1
1
end_operator
begin_operator
lift hoist0 crate1 pallet4 depot0
0
5
0 7 0 3
0 10 0 1
0 12 0 1
0 5 -1 0
0 18 11 1
1
end_operator
begin_operator
lift hoist0 crate2 crate0 depot0
0
5
0 8 0 3
0 10 0 1
0 11 -1 0
0 14 0 1
0 19 4 1
1
end_operator
begin_operator
lift hoist0 crate2 crate1 depot0
0
5
0 8 0 3
0 10 0 1
0 12 -1 0
0 14 0 1
0 19 5 1
1
end_operator
begin_operator
lift hoist0 crate2 crate3 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 15 -1 0
0 19 6 1
1
end_operator
begin_operator
lift hoist0 crate2 pallet0 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 1 -1 0
0 19 7 1
1
end_operator
begin_operator
lift hoist0 crate2 pallet1 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 2 -1 0
0 19 8 1
1
end_operator
begin_operator
lift hoist0 crate2 pallet2 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 3 -1 0
0 19 9 1
1
end_operator
begin_operator
lift hoist0 crate2 pallet3 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 4 -1 0
0 19 10 1
1
end_operator
begin_operator
lift hoist0 crate2 pallet4 depot0
0
5
0 8 0 3
0 10 0 1
0 14 0 1
0 5 -1 0
0 19 11 1
1
end_operator
begin_operator
lift hoist0 crate3 crate0 depot0
0
5
0 9 0 3
0 10 0 1
0 11 -1 0
0 15 0 1
0 20 4 1
1
end_operator
begin_operator
lift hoist0 crate3 crate1 depot0
0
5
0 9 0 3
0 10 0 1
0 12 -1 0
0 15 0 1
0 20 5 1
1
end_operator
begin_operator
lift hoist0 crate3 crate2 depot0
0
5
0 9 0 3
0 10 0 1
0 14 -1 0
0 15 0 1
0 20 6 1
1
end_operator
begin_operator
lift hoist0 crate3 pallet0 depot0
0
5
0 9 0 3
0 10 0 1
0 15 0 1
0 1 -1 0
0 20 7 1
1
end_operator
begin_operator
lift hoist0 crate3 pallet1 depot0
0
5
0 9 0 3
0 10 0 1
0 15 0 1
0 2 -1 0
0 20 8 1
1
end_operator
begin_operator
lift hoist0 crate3 pallet2 depot0
0
5
0 9 0 3
0 10 0 1
0 15 0 1
0 3 -1 0
0 20 9 1
1
end_operator
begin_operator
lift hoist0 crate3 pallet3 depot0
0
5
0 9 0 3
0 10 0 1
0 15 0 1
0 4 -1 0
0 20 10 1
1
end_operator
begin_operator
lift hoist0 crate3 pallet4 depot0
0
5
0 9 0 3
0 10 0 1
0 15 0 1
0 5 -1 0
0 20 11 1
1
end_operator
begin_operator
lift hoist1 crate0 crate1 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 12 -1 0
0 17 4 2
1
end_operator
begin_operator
lift hoist1 crate0 crate2 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 14 -1 0
0 17 5 2
1
end_operator
begin_operator
lift hoist1 crate0 crate3 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 15 -1 0
0 17 6 2
1
end_operator
begin_operator
lift hoist1 crate0 pallet0 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 1 -1 0
0 17 7 2
1
end_operator
begin_operator
lift hoist1 crate0 pallet1 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 2 -1 0
0 17 8 2
1
end_operator
begin_operator
lift hoist1 crate0 pallet2 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 3 -1 0
0 17 9 2
1
end_operator
begin_operator
lift hoist1 crate0 pallet3 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 4 -1 0
0 17 10 2
1
end_operator
begin_operator
lift hoist1 crate0 pallet4 distributor0
0
5
0 6 1 3
0 13 0 1
0 11 0 1
0 5 -1 0
0 17 11 2
1
end_operator
begin_operator
lift hoist1 crate1 crate0 distributor0
0
5
0 7 1 3
0 13 0 1
0 11 -1 0
0 12 0 1
0 18 4 2
1
end_operator
begin_operator
lift hoist1 crate1 crate2 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 14 -1 0
0 18 5 2
1
end_operator
begin_operator
lift hoist1 crate1 crate3 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 15 -1 0
0 18 6 2
1
end_operator
begin_operator
lift hoist1 crate1 pallet0 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 1 -1 0
0 18 7 2
1
end_operator
begin_operator
lift hoist1 crate1 pallet1 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 2 -1 0
0 18 8 2
1
end_operator
begin_operator
lift hoist1 crate1 pallet2 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 3 -1 0
0 18 9 2
1
end_operator
begin_operator
lift hoist1 crate1 pallet3 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 4 -1 0
0 18 10 2
1
end_operator
begin_operator
lift hoist1 crate1 pallet4 distributor0
0
5
0 7 1 3
0 13 0 1
0 12 0 1
0 5 -1 0
0 18 11 2
1
end_operator
begin_operator
lift hoist1 crate2 crate0 distributor0
0
5
0 8 1 3
0 13 0 1
0 11 -1 0
0 14 0 1
0 19 4 2
1
end_operator
begin_operator
lift hoist1 crate2 crate1 distributor0
0
5
0 8 1 3
0 13 0 1
0 12 -1 0
0 14 0 1
0 19 5 2
1
end_operator
begin_operator
lift hoist1 crate2 crate3 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 15 -1 0
0 19 6 2
1
end_operator
begin_operator
lift hoist1 crate2 pallet0 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 1 -1 0
0 19 7 2
1
end_operator
begin_operator
lift hoist1 crate2 pallet1 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 2 -1 0
0 19 8 2
1
end_operator
begin_operator
lift hoist1 crate2 pallet2 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 3 -1 0
0 19 9 2
1
end_operator
begin_operator
lift hoist1 crate2 pallet3 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 4 -1 0
0 19 10 2
1
end_operator
begin_operator
lift hoist1 crate2 pallet4 distributor0
0
5
0 8 1 3
0 13 0 1
0 14 0 1
0 5 -1 0
0 19 11 2
1
end_operator
begin_operator
lift hoist1 crate3 crate0 distributor0
0
5
0 9 1 3
0 13 0 1
0 11 -1 0
0 15 0 1
0 20 4 2
1
end_operator
begin_operator
lift hoist1 crate3 crate1 distributor0
0
5
0 9 1 3
0 13 0 1
0 12 -1 0
0 15 0 1
0 20 5 2
1
end_operator
begin_operator
lift hoist1 crate3 crate2 distributor0
0
5
0 9 1 3
0 13 0 1
0 14 -1 0
0 15 0 1
0 20 6 2
1
end_operator
begin_operator
lift hoist1 crate3 pallet0 distributor0
0
5
0 9 1 3
0 13 0 1
0 15 0 1
0 1 -1 0
0 20 7 2
1
end_operator
begin_operator
lift hoist1 crate3 pallet1 distributor0
0
5
0 9 1 3
0 13 0 1
0 15 0 1
0 2 -1 0
0 20 8 2
1
end_operator
begin_operator
lift hoist1 crate3 pallet2 distributor0
0
5
0 9 1 3
0 13 0 1
0 15 0 1
0 3 -1 0
0 20 9 2
1
end_operator
begin_operator
lift hoist1 crate3 pallet3 distributor0
0
5
0 9 1 3
0 13 0 1
0 15 0 1
0 4 -1 0
0 20 10 2
1
end_operator
begin_operator
lift hoist1 crate3 pallet4 distributor0
0
5
0 9 1 3
0 13 0 1
0 15 0 1
0 5 -1 0
0 20 11 2
1
end_operator
begin_operator
lift hoist2 crate0 crate1 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 12 -1 0
0 17 4 3
1
end_operator
begin_operator
lift hoist2 crate0 crate2 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 14 -1 0
0 17 5 3
1
end_operator
begin_operator
lift hoist2 crate0 crate3 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 15 -1 0
0 17 6 3
1
end_operator
begin_operator
lift hoist2 crate0 pallet0 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 1 -1 0
0 17 7 3
1
end_operator
begin_operator
lift hoist2 crate0 pallet1 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 2 -1 0
0 17 8 3
1
end_operator
begin_operator
lift hoist2 crate0 pallet2 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 3 -1 0
0 17 9 3
1
end_operator
begin_operator
lift hoist2 crate0 pallet3 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 4 -1 0
0 17 10 3
1
end_operator
begin_operator
lift hoist2 crate0 pallet4 distributor1
0
5
0 6 2 3
0 16 0 1
0 11 0 1
0 5 -1 0
0 17 11 3
1
end_operator
begin_operator
lift hoist2 crate1 crate0 distributor1
0
5
0 7 2 3
0 16 0 1
0 11 -1 0
0 12 0 1
0 18 4 3
1
end_operator
begin_operator
lift hoist2 crate1 crate2 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 14 -1 0
0 18 5 3
1
end_operator
begin_operator
lift hoist2 crate1 crate3 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 15 -1 0
0 18 6 3
1
end_operator
begin_operator
lift hoist2 crate1 pallet0 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 1 -1 0
0 18 7 3
1
end_operator
begin_operator
lift hoist2 crate1 pallet1 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 2 -1 0
0 18 8 3
1
end_operator
begin_operator
lift hoist2 crate1 pallet2 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 3 -1 0
0 18 9 3
1
end_operator
begin_operator
lift hoist2 crate1 pallet3 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 4 -1 0
0 18 10 3
1
end_operator
begin_operator
lift hoist2 crate1 pallet4 distributor1
0
5
0 7 2 3
0 16 0 1
0 12 0 1
0 5 -1 0
0 18 11 3
1
end_operator
begin_operator
lift hoist2 crate2 crate0 distributor1
0
5
0 8 2 3
0 16 0 1
0 11 -1 0
0 14 0 1
0 19 4 3
1
end_operator
begin_operator
lift hoist2 crate2 crate1 distributor1
0
5
0 8 2 3
0 16 0 1
0 12 -1 0
0 14 0 1
0 19 5 3
1
end_operator
begin_operator
lift hoist2 crate2 crate3 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 15 -1 0
0 19 6 3
1
end_operator
begin_operator
lift hoist2 crate2 pallet0 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 1 -1 0
0 19 7 3
1
end_operator
begin_operator
lift hoist2 crate2 pallet1 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 2 -1 0
0 19 8 3
1
end_operator
begin_operator
lift hoist2 crate2 pallet2 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 3 -1 0
0 19 9 3
1
end_operator
begin_operator
lift hoist2 crate2 pallet3 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 4 -1 0
0 19 10 3
1
end_operator
begin_operator
lift hoist2 crate2 pallet4 distributor1
0
5
0 8 2 3
0 16 0 1
0 14 0 1
0 5 -1 0
0 19 11 3
1
end_operator
begin_operator
lift hoist2 crate3 crate0 distributor1
0
5
0 9 2 3
0 16 0 1
0 11 -1 0
0 15 0 1
0 20 4 3
1
end_operator
begin_operator
lift hoist2 crate3 crate1 distributor1
0
5
0 9 2 3
0 16 0 1
0 12 -1 0
0 15 0 1
0 20 5 3
1
end_operator
begin_operator
lift hoist2 crate3 crate2 distributor1
0
5
0 9 2 3
0 16 0 1
0 14 -1 0
0 15 0 1
0 20 6 3
1
end_operator
begin_operator
lift hoist2 crate3 pallet0 distributor1
0
5
0 9 2 3
0 16 0 1
0 15 0 1
0 1 -1 0
0 20 7 3
1
end_operator
begin_operator
lift hoist2 crate3 pallet1 distributor1
0
5
0 9 2 3
0 16 0 1
0 15 0 1
0 2 -1 0
0 20 8 3
1
end_operator
begin_operator
lift hoist2 crate3 pallet2 distributor1
0
5
0 9 2 3
0 16 0 1
0 15 0 1
0 3 -1 0
0 20 9 3
1
end_operator
begin_operator
lift hoist2 crate3 pallet3 distributor1
0
5
0 9 2 3
0 16 0 1
0 15 0 1
0 4 -1 0
0 20 10 3
1
end_operator
begin_operator
lift hoist2 crate3 pallet4 distributor1
0
5
0 9 2 3
0 16 0 1
0 15 0 1
0 5 -1 0
0 20 11 3
1
end_operator
begin_operator
load hoist0 crate0 truck0 depot0
1
0 0
2
0 10 -1 0
0 17 1 0
1
end_operator
begin_operator
load hoist0 crate1 truck0 depot0
1
0 0
2
0 10 -1 0
0 18 1 0
1
end_operator
begin_operator
load hoist0 crate2 truck0 depot0
1
0 0
2
0 10 -1 0
0 19 1 0
1
end_operator
begin_operator
load hoist0 crate3 truck0 depot0
1
0 0
2
0 10 -1 0
0 20 1 0
1
end_operator
begin_operator
load hoist1 crate0 truck0 distributor0
1
0 1
2
0 13 -1 0
0 17 2 0
1
end_operator
begin_operator
load hoist1 crate1 truck0 distributor0
1
0 1
2
0 13 -1 0
0 18 2 0
1
end_operator
begin_operator
load hoist1 crate2 truck0 distributor0
1
0 1
2
0 13 -1 0
0 19 2 0
1
end_operator
begin_operator
load hoist1 crate3 truck0 distributor0
1
0 1
2
0 13 -1 0
0 20 2 0
1
end_operator
begin_operator
load hoist2 crate0 truck0 distributor1
1
0 2
2
0 16 -1 0
0 17 3 0
1
end_operator
begin_operator
load hoist2 crate1 truck0 distributor1
1
0 2
2
0 16 -1 0
0 18 3 0
1
end_operator
begin_operator
load hoist2 crate2 truck0 distributor1
1
0 2
2
0 16 -1 0
0 19 3 0
1
end_operator
begin_operator
load hoist2 crate3 truck0 distributor1
1
0 2
2
0 16 -1 0
0 20 3 0
1
end_operator
begin_operator
unload hoist0 crate0 truck0 depot0
1
0 0
2
0 10 0 1
0 17 0 1
1
end_operator
begin_operator
unload hoist0 crate1 truck0 depot0
1
0 0
2
0 10 0 1
0 18 0 1
1
end_operator
begin_operator
unload hoist0 crate2 truck0 depot0
1
0 0
2
0 10 0 1
0 19 0 1
1
end_operator
begin_operator
unload hoist0 crate3 truck0 depot0
1
0 0
2
0 10 0 1
0 20 0 1
1
end_operator
begin_operator
unload hoist1 crate0 truck0 distributor0
1
0 1
2
0 13 0 1
0 17 0 2
1
end_operator
begin_operator
unload hoist1 crate1 truck0 distributor0
1
0 1
2
0 13 0 1
0 18 0 2
1
end_operator
begin_operator
unload hoist1 crate2 truck0 distributor0
1
0 1
2
0 13 0 1
0 19 0 2
1
end_operator
begin_operator
unload hoist1 crate3 truck0 distributor0
1
0 1
2
0 13 0 1
0 20 0 2
1
end_operator
begin_operator
unload hoist2 crate0 truck0 distributor1
1
0 2
2
0 16 0 1
0 17 0 3
1
end_operator
begin_operator
unload hoist2 crate1 truck0 distributor1
1
0 2
2
0 16 0 1
0 18 0 3
1
end_operator
begin_operator
unload hoist2 crate2 truck0 distributor1
1
0 2
2
0 16 0 1
0 19 0 3
1
end_operator
begin_operator
unload hoist2 crate3 truck0 distributor1
1
0 2
2
0 16 0 1
0 20 0 3
1
end_operator
0
