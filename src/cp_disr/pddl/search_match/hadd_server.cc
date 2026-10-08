// h_add (additive delete-relaxation heuristic, unit action cost) of a grounded STRIPS task, answering batches of states over a pipe.
// Used only as a scorer of the shared search engine of card C1-DEPOTS-SEARCH-MATCH-V1.
//
// task file (same export as exact_dist):  A D G
//             init: k i1 .. ik
//             goal: g i1 .. ig
//             then A lines, one per action:  np p.. nn n.. na a.. nd d..   (pre / negative-pre / add / delete atom indices; negative preconditions and deletes are ignored: delete relaxation)
// stdin:  h K            followed by K lines   n i1 .. in   (true atoms of a state)
// stdout: K lines, h_add of each state as an integer, or -1 when some goal atom is unreachable in the relaxation (the caller sorts those last)
//         quit
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace std;
typedef long long ll;
static const ll INF = (ll)4e18;

int A, N, G;
vector<vector<int>> pre_, add_, by_pre;
vector<int> goal_, nopre;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: hadd_server task.txt\n"); return 2; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open task\n"); return 2; }
    if (fscanf(f, "%d %d %d", &A, &N, &G) != 3) return 2;
    char tag[32]; int k, x;
    if (fscanf(f, "%31s %d", tag, &k) != 2) return 2;
    for (int i = 0; i < k; i++) if (fscanf(f, "%d", &x) != 1) return 2;      // initial state ignored: states come from stdin
    if (fscanf(f, "%31s %d", tag, &k) != 2) return 2;
    for (int i = 0; i < k; i++) { if (fscanf(f, "%d", &x) != 1) return 2; goal_.push_back(x); }
    pre_.resize(A); add_.resize(A); by_pre.resize(N);
    for (int a = 0; a < A; a++) {
        for (int j = 0; j < 4; j++) {
            if (fscanf(f, "%d", &k) != 1) return 2;
            for (int i = 0; i < k; i++) {
                if (fscanf(f, "%d", &x) != 1) return 2;
                if (j == 0) pre_[a].push_back(x);
                else if (j == 2) add_[a].push_back(x);
            }
        }
        if (pre_[a].empty()) nopre.push_back(a);
        for (int p : pre_[a]) by_pre[p].push_back(a);
    }
    fclose(f);
    vector<ll> cost(N), acost(A);
    vector<int> cnt(A);
    vector<char> is_goal(N, 0), settled(N, 0);
    for (int g : goal_) is_goal[g] = 1;
    string cmd;
    typedef pair<ll, int> P;
    while (cin >> cmd) {
        if (cmd == "quit") break;
        if (cmd != "h") continue;
        int K; cin >> K;
        for (int q = 0; q < K; q++) {
            int n; cin >> n;
            fill(cost.begin(), cost.end(), INF);
            fill(acost.begin(), acost.end(), 0);
            fill(settled.begin(), settled.end(), 0);
            for (int a = 0; a < A; a++) cnt[a] = (int)pre_[a].size();
            priority_queue<P, vector<P>, greater<P>> pq;
            for (int i = 0; i < n; i++) { cin >> x; if (cost[x] != 0) { cost[x] = 0; pq.push(P(0, x)); } }
            for (int a : nopre) for (int ad : add_[a]) if (1 < cost[ad]) { cost[ad] = 1; pq.push(P(1, ad)); }
            int remaining = 0;
            for (int g : goal_) remaining++;
            // goals may repeat? exported goal set has distinct atoms
            while (!pq.empty() && remaining > 0) {
                P top = pq.top(); pq.pop();
                int p = top.second;
                if (settled[p] || top.first > cost[p]) continue;
                settled[p] = 1;
                if (is_goal[p]) remaining--;
                for (int a : by_pre[p]) {
                    acost[a] += top.first;
                    if (--cnt[a] == 0) {
                        ll c = acost[a] + 1;
                        for (int ad : add_[a]) if (c < cost[ad]) { cost[ad] = c; pq.push(P(c, ad)); }
                    }
                }
            }
            ll h = 0; bool inf = false;
            for (int g : goal_) { if (cost[g] >= INF) { inf = true; break; } h += cost[g]; }
            printf("%lld\n", inf ? -1LL : h);
        }
        fflush(stdout);
    }
    return 0;
}
