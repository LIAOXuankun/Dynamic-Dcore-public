
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <utility>

#include "Repeel.h"


/**
    find neighbor
**/
void Repeel::findNeib(vector<ArrayEntry> &vAdj1, vector<ArrayEntry> &vAdj2, vector<pair<uint32_t, uint32_t>> & tris)
{
    size_t p1 = 0, p2 = 0;
    while (p1 < vAdj1.size() && p2 < vAdj2.size()) {
        if (vAdj1[p1].vid == vAdj2[p2].vid) {
            tris.push_back({vAdj1[p1].eid, vAdj2[p2].eid});
            ++p1; ++p2;
        } else if (vAdj1[p1].vid < vAdj2[p2].vid) {
            ++p1;
        } else {
            ++p2;
        }
    }
}
/**
 * complete partial d-core that only contains k value of unique k-lists
 */
void Repeel::completePartialDcore(vector<vector<pair<::uint32_t, uint32_t>>> &d_cores) {
    vector<vector<pair<::uint32_t,uint32_t>>> complete_d_cores;

    uint32_t cnt = 0;
    for(auto d_core : d_cores){
        ASSERT_MSG(d_core.size() >= 1, "d_core size error: " << d_core.size());
        vector<pair<uint32_t,uint32_t>> complete_d_core;
        if(d_core.size() == 1){
            complete_d_core.emplace_back(d_core[0].first,d_core[0].second);
            while (complete_d_core.back().first > 0) {
                complete_d_core.emplace_back(complete_d_core.back().first  - 1, d_core[0].second);
                ASSERT_MSG(complete_d_core.back().first < n_ , "wrong complete d-core: " << complete_d_core.back().first);
                if(complete_d_core.back().first == 0){
                    break;
                }
            }
        
        }
        else{
            for(::uint32_t i = 0; i < d_core.size() ; ++i){
                if(i == 0 || ((complete_d_core.back().first - d_core[i].first) == 1)){
                    complete_d_core.emplace_back(d_core[i].first, d_core[i].second);
                }
                else{
                    uint32_t loop_cnt = 0;
                    while ((complete_d_core.back().first - d_core[i].first) > 1){
                        ASSERT_MSG(complete_d_core.back().first < n_ , "wrong complete d-core: " << complete_d_core.back().first);
                        complete_d_core.push_back(make_pair(complete_d_core.back().first - 1, complete_d_core.back().second));
                        if(complete_d_core.back().first - d_core[i].first == 1) {
                            complete_d_core.emplace_back(d_core[i].first , d_core[i].second);
                            break;
                        }
                        ++loop_cnt;
                        ASSERT_MSG(loop_cnt <= 1000, " wrong: " << complete_d_core.back().first << " " << d_core[i].first);
                    }
                }
            }
        }
        // handle those cases when the last k value is not 0
        while (complete_d_core.back().first != 0) {
            complete_d_core.emplace_back(complete_d_core.back().first  - 1, d_core.back().second);
        }
        complete_d_cores.push_back(complete_d_core);

        ++cnt;
    }

    d_cores = complete_d_cores;
}
/**
    init
**/
Repeel::Repeel(vector<pair<uint32_t, uint32_t>> &vEdges)
{
    edges_ = vEdges;
    sort(edges_.begin(), edges_.end());
    auto it = unique(edges_.begin(), edges_.end());
    edges_.erase(it, edges_.end());    // deduplication


    //build the adj list
    m_ = edges_.size();
    nodes_.reserve(2 * m_);
    map<uint32_t, vector<pair<uint32_t, uint32_t> > > mpRec;
    map<uint32_t, vector<pair<uint32_t, uint32_t> > >::iterator itmpRec;
    vector<pair<uint32_t, uint32_t> >::iterator itvE;
    for (uint32_t eid = 0; eid < m_; ++eid)
    {
        const uint32_t x = edges_[eid].first;
        const uint32_t y = edges_[eid].second;

        mpRec[x].emplace_back(eid, 1);
        mpRec[y].emplace_back(eid, 2);
    }
    int iRePid = 0, max_id = 0;
    for (itmpRec = mpRec.begin(); itmpRec != mpRec.end(); ++itmpRec, ++iRePid)
    {
        nodes_.push_back(itmpRec->first);
        if (itmpRec->first > max_id)
        {
            max_id = itmpRec->first;
        }
        for (itvE = itmpRec->second.begin(); itvE != itmpRec->second.end(); ++itvE)
        {
            int iEid = itvE->first;
            if (1 == itvE->second)
            {
                edges_[iEid].first = iRePid;
            }
            else if (2 == itvE->second)
            {
                edges_[iEid].second = iRePid;
            }
        }
    }
    n_ = nodes_.size();
    //printf("PEEL init n: %d, m: %d\n", n_, m_);
}
/**
 peel original graph to get k-lists, returned the deduplicated k-lists ((k,0)-cores)
**/
void    Repeel::peelKlist(vector<vector<pair<uint32_t, uint32_t>>> &independent_k_lists)
{
    // initialize adjacency arrays
    adj_in.resize(n_);
    adj_out.resize(n_);
    for (uint32_t eid = 0; eid < m_; ++eid) {
        const uint32_t v1 = edges_[eid].first;
        const uint32_t v2 = edges_[eid].second;
        adj_out[v1].push_back({v2, eid});
        adj_in[v2].push_back({v1, eid});
    }
    for (uint32_t vid = 0; vid < n_; ++vid) {
        std::sort(adj_in[vid].begin(), adj_in[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
        std::sort(adj_out[vid].begin(), adj_out[vid].end(),
                  [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                      return ae1.vid < ae2.vid;
                  });
    }

    // bin-sort peeling for D-core w.r.t the in-degree, get all the unique k-lists [(k,0)-core]
    // 1.initialization
    std::vector<::uint32_t> deg_in(n_, 0);
    //std::vector<::uint32_t> in_coreness(n_, 0);
    in_coreness.resize(n_);
    std::vector<bool> deleted(m_, false);  //the edge is deleted or not?

    // 2. get the in-degree
    ::uint32_t max_in_deg = 0;
    for (uint32_t vid : nodes_) {
        deg_in[vid] = adj_in[vid].size();
        ASSERT_MSG(deg_in[vid] < n_, "wrong in-degree error " << deg_in[vid] << " " << n_ << " " << vid);
        max_in_deg = std::max(max_in_deg, deg_in[vid]);
    }

    // 3. initialize buckets
    vector<unordered_set<::uint32_t>> buckets(max_in_deg + 1);
    for(::uint32_t vid : nodes_) {
        ASSERT_MSG(deg_in[vid] <= n_, "wrong in-degree error");
        buckets[deg_in[vid]].insert(vid);
    }
    // 4. peel vertices with in-degree less than k
    uint32_t max_in_coreness = 0;
    for (::uint32_t k = 0; k < buckets.size(); ++k) {
        while (!buckets[k].empty()) {
            uint32_t vid = *buckets[k].begin();
            buckets[k].erase(buckets[k].begin());
            in_coreness[vid] = k;
            max_in_coreness = std::max(max_in_coreness, in_coreness[vid]);
            for (const ArrayEntry &ae: adj_out[vid]) {   //for out neighbors, decrease their in-degree
                ::uint32_t nbr = ae.vid;
                if(!deleted[ae.eid]){
                    deleted[ae.eid] = true;
                    if(deg_in[nbr] > k) {
                        buckets[deg_in[nbr]].erase(nbr);
                        --deg_in[nbr];
                        buckets[deg_in[nbr]].insert(nbr);
                    }
                }
            }
            for (const ArrayEntry &ae: adj_in[vid]) {  //delete incoming edges of deleted vertex: vid
                if (!(deleted[ae.eid])) {
                    deleted[ae.eid] = true;
                }
            }
        }
    }
    //printf("max_in_coreness: %u\n", max_in_coreness);
    // 5. get the independent k-lists
    auto non_independent_k_lists = vector<vector<pair<uint32_t, uint32_t> >>(max_in_coreness + 1);
    // 5.1 each row is a k-list, consisting of edges in this (k,0)-core
    for(::uint32_t vid : nodes_) {
        ASSERT_MSG(in_coreness[vid] >= 0, "in_coreness < 0 error");
        for (const ArrayEntry& ae : adj_out[vid]) {
            ::uint32_t nbr = ae.vid;
            if (in_coreness[nbr] >= in_coreness[vid]) {
                non_independent_k_lists[in_coreness[vid]].emplace_back(vid, nbr);
            }
            else{
                non_independent_k_lists[in_coreness[nbr]].emplace_back(vid, nbr);
            }
        }
    }
    // 5.2 get the independent k-lists
    //std::printf("max_in_coreness: %d\n", max_in_coreness);
    for(int i = max_in_coreness ; i > 0; --i){
        vector<pair<::uint32_t,::uint32_t>> tmp;
        tmp.emplace_back(i,0);
        tmp.insert(tmp.end(), non_independent_k_lists[i].begin(), non_independent_k_lists[i].end());
        if(independent_k_lists.empty() || ( independent_k_lists.back().size() != tmp.size())) {
            independent_k_lists.push_back(tmp);
        }
        non_independent_k_lists[i-1].insert(non_independent_k_lists[i-1].end(), non_independent_k_lists[i].begin(),
                                            non_independent_k_lists[i].end());
    }
    vector<pair<::uint32_t,::uint32_t>> zero_zero_core;
    zero_zero_core.emplace_back(0,0);
    zero_zero_core.insert(zero_zero_core.end(), non_independent_k_lists[0].begin(), non_independent_k_lists[0].end());
    if(max_in_coreness == 0){
        independent_k_lists = non_independent_k_lists;
    }
    else{
        if(zero_zero_core.size() != independent_k_lists.back().size()){
            independent_k_lists.push_back(zero_zero_core);
        }
    }

    unique_k_lists = independent_k_lists;  //each row: (k,0), (u1,v1), (u2,v2), ... unique_k_lists[i+1][0].first < unique_k_lists[i][0].first
    non_independent_k_lists.clear();
}
/**
 peel independent k-lists (k,0)-cores, returned the D-core decomposition results
**/
void Repeel::peelDcore(vector<vector<pair<uint32_t, uint32_t>>> &d_core_decomposition) {
    d_core_decomposition.resize(n_);
    uint32_t max_out_coreness = 0;

    // iterate through all the unique k-lists and peel them,
    // from the one with the largest k value to the one with the smallest k value
    for(auto k_list : unique_k_lists){
        // initialize adjacency arrays for the current k-list
        ::uint32_t current_k = k_list[0].first;
        vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
        // 1 map the node id in k-list -> consistent, zero-based node id
        set<::uint32_t> node_recorder;
        map<::uint32_t,uint32_t> node_map, node_map_reverse;
        for (uint32_t eid = 1; eid < k_list.size(); ++eid) {
            const uint32_t v1 = k_list[eid].first;
            const uint32_t v2 = k_list[eid].second;
            node_recorder.insert(v1);
            node_recorder.insert(v2);
        }
        uint32_t node_recorder_cnt = 0;
        for (uint32_t vid : node_recorder) {
            if(optimize){
                node_map[vid] = node_recorder_cnt;
            }
            else{
                node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));   
            }
            
            node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
            ++node_recorder_cnt;
        }
        ::uint32_t tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
        ::uint32_t tmp_m_ = k_list.size() - 1;          //the number of edges in the current k-list
        k_adj_in.resize(tmp_n_);
        k_adj_out.resize(tmp_n_);
        // 2 initialize the adjacency arrays
        for (uint32_t eid = 1; eid < k_list.size(); ++eid) {   //note that the first element is (k,0)
            const uint32_t v1 = k_list[eid].first;
            const uint32_t v2 = k_list[eid].second;
            k_adj_out[node_map[v1]].push_back({node_map[v2], eid - 1});
            k_adj_in[node_map[v2]].push_back({node_map[v1], eid - 1});
        }
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                      [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                          return ae1.vid < ae2.vid;
                      });
            std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                      [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                          return ae1.vid < ae2.vid;
                      });
        }
        // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
        // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
        // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
        // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

        // 1.initialization
        std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
        std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);
        std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
        // 2. get the in-degree
        ::uint32_t max_out_deg = 0;
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            deg_out[vid] = k_adj_out[vid].size();
            max_out_deg = std::max(max_out_deg, deg_out[vid]);
            deg_in[vid] = k_adj_in[vid].size();
        }

        // 3. initialize buckets
        vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
            buckets[deg_out[vid]].insert(vid);
        }
        // 4. peel vertices with out-degree less than k
        for (::uint32_t k = 0; k < buckets.size(); ++k) {
            if(buckets[k].empty()) continue;
            while (!buckets[k].empty()) {
                uint32_t vid = *buckets[k].begin();
                buckets[k].erase(buckets[k].begin());
                sub_out_coreness[vid] = k;
                max_out_coreness = std::max(max_out_coreness, sub_out_coreness[vid]);
                for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                    ::uint32_t nbr = ae.vid;
                    if(!(deleted[ae.eid])){
                        deleted[ae.eid] = true;
                        if(deg_out[nbr] > k && deg_in[nbr] >= current_k) {
                            buckets[deg_out[nbr]].erase(nbr);
                            --deg_out[nbr];
                            buckets[deg_out[nbr]].insert(nbr);
                        }
                    }
                }
                for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                    ::uint32_t nbr = ae.vid;
                    if (!(deleted[ae.eid])) {
                        deleted[ae.eid] = true;
                        --deg_in[nbr];
                        if(deg_in[nbr] < current_k){
                            ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                            buckets[deg_out[nbr]].erase(nbr);
                            buckets[k].insert(nbr);
                        }
                    }
                }
            }
        }
        //record the (current_k, a_{u}^{current_k}) pair for all vertices
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            uint32_t u = node_map_reverse[vid];
            d_core_decomposition[u].emplace_back(make_pair(current_k, sub_out_coreness[vid]));
        }
    }
    //complete the d-core decomposition result by adding the (k,l) pair for all k values
    completePartialDcore(d_core_decomposition);
    //printf("max_outcoreness: %d", max_out_coreness);
}
/**
 * the advanced peeling algorithm for D-core decomposition, prune the search space based on the in-coreness of modified edges
**/
void Repeel::optimizedPeelDcore(const vector<pair<::uint32_t, ::uint32_t>>& modified_edge,
                                map<::uint32_t,uint32_t> &new_to_old_node_map,
                                bool b_insertion,
                                vector<vector<pair<uint32_t, uint32_t>>> &new_d_core_decomposition,
                                vector<vector<pair<uint32_t, uint32_t>>> &old_d_core_decomposition) {
    vector<vector<pair<::uint32_t,::uint32_t>>> new_independent_k_lists;  //the new independent k-lists after the edge modification
    peelKlist(new_independent_k_lists); //get the new independent k-lists

    // 1.iterate through the inserted/deleted edges to get the new in-coreness for both vertex
    uint32_t min_incoreness = 1;

    for(auto edge : modified_edge){
        if(old_d_core_decomposition.size() != n_){
            min_incoreness = 1;
        }
        else{
            uint32_t tmp_min_incoreness = std::min(in_coreness[edge.first], in_coreness[edge.second]);
            min_incoreness = std::max(min_incoreness, tmp_min_incoreness);
        }
    }
    printf("optimized repeel min_incoreness:%d \n", min_incoreness);


//    pair<::uint32_t,uint32_t> new_edge = modified_edge[0];
//    ::uint32_t min_incoreness = 0, incoreness_u = 0, incoreness_v = 0;
//    //    An edge is deleted from original graph, resulting the deletion of a vertex.
//    //    In this case, we directedly set the min_incoreness to 1
//    if(old_d_core_decomposition.size() != n_){
//        min_incoreness = 1;
//    }
//    else{
//        min_incoreness = std::min(in_coreness[new_edge.first], in_coreness[new_edge.second]);
//    }

    //2. obtain the k-lists needed to be update based on min_coreness
    vector<vector<pair<::uint32_t,::uint32_t>>> to_be_update_k_lists, partial_d_core_decomposition;
    partial_d_core_decomposition.resize(n_);
    for(auto k_list : new_independent_k_lists){
        if(k_list[0].first <= min_incoreness + 1){
            to_be_update_k_lists.push_back(k_list);
        }
    }

    //3. peel the k-lists in to_be_update_k_lists
    for (vector<pair<::uint32_t,::uint32_t>> k_list : to_be_update_k_lists){
        // initialize adjacency arrays for the current k-list
        ::uint32_t current_k = k_list[0].first;
        vector<vector<ArrayEntry>> k_adj_in, k_adj_out;
        // 3.1 map the node id in k-list -> consistent, zero-based node id
        set<::uint32_t> node_recorder;
        map<::uint32_t,uint32_t> node_map, node_map_reverse;
        for (uint32_t eid = 1; eid < k_list.size(); ++eid) {
            const uint32_t v1 = k_list[eid].first;
            const uint32_t v2 = k_list[eid].second;
            node_recorder.insert(v1);
            node_recorder.insert(v2);
        }
        uint32_t node_recorder_cnt = 0;
        for (uint32_t vid : node_recorder) {
            if(optimize){
                node_map[vid] = node_recorder_cnt;
            }
            else{
                node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
            }
            //node_map[vid] = std::distance(node_recorder.begin(), node_recorder.find(vid));
            node_map_reverse[node_map[vid]] = vid;    //map the consistent, zero-based node id -> node id in k-list
            ++node_recorder_cnt;
        }
        ::uint32_t tmp_n_ = node_recorder.size();       //the number of vertices in the current k-list
        ::uint32_t tmp_m_ = k_list.size() - 1;          //the number of edges in the current k-list
        k_adj_in.resize(tmp_n_);
        k_adj_out.resize(tmp_n_);
        // 3.2 initialize the adjacency arrays
        for (uint32_t eid = 1; eid < k_list.size(); ++eid) {   //note that the first element is (k,0)
            const uint32_t v1 = k_list[eid].first;
            const uint32_t v2 = k_list[eid].second;
            k_adj_out[node_map[v1]].push_back({node_map[v2], eid - 1});
            k_adj_in[node_map[v2]].push_back({node_map[v1], eid - 1});
        }
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            std::sort(k_adj_out[vid].begin(), k_adj_out[vid].end(),
                      [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                          return ae1.vid < ae2.vid;
                      });
            std::sort(k_adj_in[vid].begin(), k_adj_in[vid].end(),
                      [](const ArrayEntry& ae1, const ArrayEntry& ae2) {
                          return ae1.vid < ae2.vid;
                      });
        }
        // bin-sort peeling for k-list w.r.t the out-degree, get [(k,l)-core] for all the l values
        // note that when we delete v, for its in-neighbor w, we maintain their sub_out_coreness(a_{w}^{k})
        // for its out-neighbor u, we maintain their in_degree in current k-list, if the in_degree < k after v's deletion
        // then we have a_{v}^{k} = a_{u}^{k}, and we delete u as well.

        // 3.1.initialization
        std::vector<::uint32_t> deg_in(tmp_n_, 0), deg_out(tmp_n_, 0);
        std::vector<::uint32_t> sub_out_coreness(tmp_n_, 0);

        std::vector<bool> deleted(tmp_m_, false);  //the edge is deleted or not?
        // 3.2. get the in-degree
        ::uint32_t max_out_deg = 0;
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            deg_out[vid] = k_adj_out[vid].size();
            max_out_deg = std::max(max_out_deg, deg_out[vid]);
            deg_in[vid] = k_adj_in[vid].size();
        }
        // 3.3. initialize buckets
        vector<unordered_set<::uint32_t>> buckets(max_out_deg + 1);
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            ASSERT_MSG(deg_out[vid] <= max_out_deg, "wrong sub-out-degree error " << current_k << " vid " << vid << " " << deg_out[vid] << " " << max_out_deg);
            buckets[deg_out[vid]].insert(vid);
        }
        // 3.4. peel vertices with out-degree less than k
        for (::uint32_t k = 0; k < buckets.size(); ++k) {
            while (!buckets[k].empty()) {
                uint32_t vid = *buckets[k].begin();
                buckets[k].erase(buckets[k].begin());
                sub_out_coreness[vid] = k;
                //max_out_coreness = std::max(max_out_coreness, sub_out_coreness[vid]);
                for (const ArrayEntry &ae: k_adj_in[vid]) {   //for in neighbors, decrease their out-degree
                    ::uint32_t nbr = ae.vid;
                    if (deg_out[nbr] > k && !(deleted[ae.eid]) && deg_in[nbr] >= current_k) {  // edge is not deleted yet
                        deleted[ae.eid] = true;
                        buckets[deg_out[nbr]].erase(nbr);
                        --deg_out[nbr];
                        buckets[deg_out[nbr]].insert(nbr);
                    }
                }
                for (const ArrayEntry &ae: k_adj_out[vid]) {  //for out neighbors, check their in-degree to see if it >= k
                    ::uint32_t nbr = ae.vid;
                    if (!(deleted[ae.eid])) {
                        deleted[ae.eid] = true;
                        --deg_in[nbr];
                        if(deg_in[nbr] < current_k){
                            ASSERT_MSG(deg_out[nbr] >= k, "wrong out degree error " << k << " vid " << nbr << " " << deg_out[nbr] << " " << k);
                            if(deg_out[nbr] > k){
                                buckets[deg_out[nbr]].erase(nbr);
                                buckets[k].insert(nbr);
                            }
                        }
                    }
                }
            }
        }

        // record the (current_k, a_{u}^{current_k}) pair for all vertices
        for (uint32_t vid = 0; vid < tmp_n_; ++vid) {
            uint32_t u = node_map_reverse[vid];
            partial_d_core_decomposition[u].emplace_back(current_k, sub_out_coreness[vid]);
        }

    }

    //4. combine the new d_core_decomposition result with the old ones
    new_d_core_decomposition.clear();
    new_d_core_decomposition.resize(n_);
    completePartialDcore(partial_d_core_decomposition);

    for(uint32_t vid = 0; vid < n_; ++vid){
        if(partial_d_core_decomposition[vid].empty()){
            new_d_core_decomposition[vid] = old_d_core_decomposition[/*new_to_old_node_map[vid]*/vid];
            continue;
        }
        else{
            //insertion case
            if(b_insertion){
                // insertion case, new_d_core_decomposition of vid has an incoreness >= original one
                if(old_d_core_decomposition[/*new_to_old_node_map[vid]*/vid][0].first <= partial_d_core_decomposition[vid][0].first){
                    new_d_core_decomposition[vid] = partial_d_core_decomposition[vid];
                    continue;
                }
                else{
                    for(auto pair: old_d_core_decomposition[/*new_to_old_node_map[vid]*/vid]){
                        if(pair.first > partial_d_core_decomposition[vid][0].first){
                            new_d_core_decomposition[vid].emplace_back(pair);
                        }
                        else{
                            new_d_core_decomposition[vid].insert(new_d_core_decomposition[vid].end(), partial_d_core_decomposition[vid].begin(), partial_d_core_decomposition[vid].end());
                            break;
                        }
                    }
                }
            }
            //deletion case
            else{
                if(in_coreness[vid] == partial_d_core_decomposition[vid][0].first){
                    new_d_core_decomposition[vid] = partial_d_core_decomposition[vid];
                    continue;
                }
                else{
                    for(auto pair: old_d_core_decomposition[/*new_to_old_node_map[vid]*/vid]){
                        if(pair.first > in_coreness[vid]) continue;
                        if(pair.first <= in_coreness[vid] &&
                                pair.first > partial_d_core_decomposition[vid][0].first){
                            new_d_core_decomposition[vid].emplace_back(pair);
                            continue;
                        }
                        else if(pair.first == partial_d_core_decomposition[vid][0].first){
                            new_d_core_decomposition[vid].insert(new_d_core_decomposition[vid].end(), partial_d_core_decomposition[vid].begin(), partial_d_core_decomposition[vid].end());
                            break;
                        }
                    }
                }
            }

        }
    }

}
