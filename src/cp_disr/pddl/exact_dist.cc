// Exact unit-cost distance-to-goal over the complete reachable state space of a grounded STRIPS task (bit-set states).
// Used only to produce training / development labels (optimal action sets, exact successor costs) and a canonical optimal plan for small tasks; evaluation-time policies never call it.
//
// task file:  A D G
//             init: k i1 .. ik
//             goal: g i1 .. ig
//             then A lines, one per action:  np p.. nn n.. na a.. nd d..      (pre / negative-pre / add / delete atom indices)
// stdin commands:  build <maxStates>        -> OK <states> <goal_states> <max_dist>   |  TOO_BIG <states_so_far>
//                  q k i1..ik               -> <dist|-1> <m> a1 d1 a2 d2 ...   (dist of the state, then every applicable action with the dist of its successor; -1 = unknown / unreachable state)
//                  plan                     -> canonical optimal plan from init (lowest action index among optimal actions at every step), or NOPLAN
//                  quit
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
typedef array<uint64_t, 4> State;
struct SH {
    size_t operator()(const State& s) const {
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 4; i++) { h ^= s[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); h *= 1099511628211ULL; }
        return (size_t)h;
    }
};
struct Act { State pre, neg, add, del; };

static inline void setbit(State& s, int i) { s[i >> 6] |= (1ULL << (i & 63)); }
static inline bool subset(const State& a, const State& b) { for (int i = 0; i < 4; i++) if (a[i] & ~b[i]) return false; return true; }
static inline bool disjoint(const State& a, const State& b) { for (int i = 0; i < 4; i++) if (a[i] & b[i]) return false; return true; }

int A, D, G;
vector<Act> acts;
State init_s, goal_s;
vector<State> states;
unordered_map<State, uint32_t, SH> idx;
vector<int16_t> dist;
bool built = false;

static inline State apply_act(const State& s, const Act& a) {
    State t;
    for (int i = 0; i < 4; i++) t[i] = (s[i] & ~a.del[i]) | a.add[i];
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: exact_dist task.txt\n"); return 2; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open task\n"); return 2; }
    if (fscanf(f, "%d %d %d", &A, &D, &G) != 3) return 2;
    if (D > 256) { fprintf(stderr, "too many atoms\n"); return 3; }
    init_s.fill(0); goal_s.fill(0);
    char tag[32]; int k, x;
    if (fscanf(f, "%31s %d", tag, &k) != 2) return 2;
    for (int i = 0; i < k; i++) { if (fscanf(f, "%d", &x) != 1) return 2; setbit(init_s, x); }
    if (fscanf(f, "%31s %d", tag, &k) != 2) return 2;
    for (int i = 0; i < k; i++) { if (fscanf(f, "%d", &x) != 1) return 2; setbit(goal_s, x); }
    acts.resize(A);
    for (int a = 0; a < A; a++) {
        State* ss[4] = {&acts[a].pre, &acts[a].neg, &acts[a].add, &acts[a].del};
        for (int j = 0; j < 4; j++) {
            ss[j]->fill(0);
            if (fscanf(f, "%d", &k) != 1) return 2;
            for (int i = 0; i < k; i++) { if (fscanf(f, "%d", &x) != 1) return 2; setbit(*ss[j], x); }
        }
    }
    fclose(f);
    string line;
    while (getline(cin, line)) {
        istringstream is(line);
        string cmd; is >> cmd;
        if (cmd == "quit") break;
        if (cmd == "build") {
            long long maxs = 30000000; is >> maxs;
            states.clear(); idx.clear(); dist.clear();
            states.push_back(init_s); idx[init_s] = 0;
            vector<uint32_t> from, to;
            bool big = false;
            for (size_t u = 0; u < states.size() && !big; u++) {
                State s = states[u];
                for (int a = 0; a < A; a++) {
                    if (!subset(acts[a].pre, s) || !disjoint(acts[a].neg, s)) continue;
                    State t = apply_act(s, acts[a]);
                    auto it = idx.find(t);
                    uint32_t v;
                    if (it == idx.end()) {
                        if ((long long)states.size() >= maxs) { big = true; break; }
                        v = states.size(); states.push_back(t); idx[t] = v;
                    } else v = it->second;
                    if (v != (uint32_t)u) { from.push_back((uint32_t)u); to.push_back(v); }
                }
            }
            if (big) { printf("TOO_BIG %zu\n", states.size()); fflush(stdout); built = false; states.clear(); idx.clear(); continue; }
            size_t n = states.size();
            // reverse CSR
            vector<uint32_t> start(n + 1, 0);
            for (size_t e = 0; e < to.size(); e++) start[to[e] + 1]++;
            for (size_t i = 0; i < n; i++) start[i + 1] += start[i];
            vector<uint32_t> pos(start.begin(), start.end() - 1), radj(to.size());
            for (size_t e = 0; e < to.size(); e++) radj[pos[to[e]]++] = from[e];
            from.clear(); from.shrink_to_fit(); to.clear(); to.shrink_to_fit();
            dist.assign(n, -1);
            vector<uint32_t> q; q.reserve(n);
            size_t ngoal = 0;
            for (size_t i = 0; i < n; i++) if (subset(goal_s, states[i])) { dist[i] = 0; q.push_back((uint32_t)i); ngoal++; }
            int maxd = 0;
            for (size_t h = 0; h < q.size(); h++) {
                uint32_t v = q[h];
                for (uint32_t e = start[v]; e < start[v + 1]; e++) {
                    uint32_t u = radj[e];
                    if (dist[u] < 0) { dist[u] = dist[v] + 1; if (dist[u] > maxd) maxd = dist[u]; q.push_back(u); }
                }
            }
            built = true;
            printf("OK %zu %zu %d\n", n, ngoal, maxd); fflush(stdout);
        } else if (cmd == "q") {
            if (!built) { printf("-1 0\n"); fflush(stdout); continue; }
            int kk; is >> kk; State s; s.fill(0);
            for (int i = 0; i < kk; i++) { int z; is >> z; setbit(s, z); }
            auto it = idx.find(s);
            if (it == idx.end()) { printf("-1 0\n"); fflush(stdout); continue; }
            ostringstream os; int m = 0;
            for (int a = 0; a < A; a++) {
                if (!subset(acts[a].pre, s) || !disjoint(acts[a].neg, s)) continue;
                State t = apply_act(s, acts[a]);
                auto jt = idx.find(t);
                os << " " << a << " " << (jt == idx.end() ? -1 : (int)dist[jt->second]);
                m++;
            }
            printf("%d %d%s\n", (int)dist[it->second], m, os.str().c_str()); fflush(stdout);
        } else if (cmd == "plan") {
            if (!built || dist[0] < 0) { printf("NOPLAN\n"); fflush(stdout); continue; }
            State s = init_s; uint32_t u = 0; ostringstream os; int guard = 0;
            while (dist[u] > 0 && guard++ < 10000) {
                bool moved = false;
                for (int a = 0; a < A; a++) {
                    if (!subset(acts[a].pre, s) || !disjoint(acts[a].neg, s)) continue;
                    State t = apply_act(s, acts[a]);
                    auto jt = idx.find(t);
                    if (jt != idx.end() && dist[jt->second] == dist[u] - 1) { os << a << " "; s = t; u = jt->second; moved = true; break; }
                }
                if (!moved) break;
            }
            printf("%s\n", os.str().c_str()); fflush(stdout);
        } else { printf("ERR\n"); fflush(stdout); }
    }
    return 0;
}

