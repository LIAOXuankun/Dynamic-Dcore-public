#include "Common.h"
#include "Repeel.h"
#include "Io.h"
#include "edgeGenerator.h"
#include "dfsSearch.h"
#include "hIndex.h"




int main(int argc, char** argv) {
    //read the directed graph and do the decomposition
    //ASSERT_MSG(2 < argc, "wrong parameters");

/**
    1. setting parameters
**/
    int batch_size = 200;      //the batch size should = num_insertion_edges = num_deletion_edges
    std::sscanf(argv[1],"%d", &batch_size);
    int num_insertion_edges = batch_size;
    int num_deletion_edges = batch_size;
    int lmax_number_of_threads = 8; //for single edge maintenance, parallel maintain multiple lmax value of multiple k-lists

    /*input graph file*/
    string file_path = "/home/comp/Dcore-maintain/data/";
    string file_name = /*"message"*/argv[2];
    string input_file_suffix = ".txt";
    string input_file_string= file_path + file_name + "/" + file_name + input_file_suffix;
    char *input_file = (char*)input_file_string.c_str();
    /*input graph decomposition file*/
    string decomposition_file_suffix = "-dcore.txt";
    string decomposition_file_string = file_path + file_name + "/" + file_name + decomposition_file_suffix;
    char *decomposition_result_file = (char*)decomposition_file_string.c_str();
    /*input graph insert edge file*/
    char *bi_graph_file = "/Users/Exp-data/dcore-maintain/email/email-bi.txt";   //transform the directed graph to bi-directed graph
    string generated_insert_file_suffix = "-insert.txt";
    string generated_insert_file_string = file_path + file_name + "/" + file_name + generated_insert_file_suffix;
    char *generated_insert_file = (char*)generated_insert_file_string.c_str();
    /*input graph delete edge file*/
    string generated_delete_file_suffix = "-delete.txt";
    string generated_delete_file_string = file_path + file_name + "/" + file_name + generated_delete_file_suffix;
    char *generated_delete_file = (char*)generated_delete_file_string.c_str();

    bool generate_modefiy_edges = false;    //whether generate edges to be inserted / deleted
    bool generate_subgraphs = false;    //whether generate subgraphs

    bool use_h_index = true;  //use h-index method / peeling method to process the (M_, 0)-core, always to be true
    bool kmax_hierarchy = true;  //parallel strategy: kmax_hierarchy
    bool kedge_set = true;     //parallel strategy: kedge_set

    bool k0core_pruning = true;
    bool reuse_pruning = true;
    bool skip_pruning = false;

    bool decomp = true;   //whether use the naive re-decomp maintenance
    bool peel = true;     //whether use the optimized re-peel maintenance
    bool dfs = false;
    std::sscanf(argv[3],"%d", &lmax_number_of_threads);
   istringstream(argv[4]) >> boolalpha >> use_h_index;
   istringstream(argv[5]) >> boolalpha >> kmax_hierarchy;
   istringstream(argv[6]) >> boolalpha >> kedge_set;
   istringstream(argv[7]) >> boolalpha >> decomp;
   istringstream(argv[8]) >> boolalpha >> peel;
   istringstream(argv[9]) >> boolalpha >> k0core_pruning;
   istringstream(argv[10]) >> boolalpha >> reuse_pruning;
   istringstream(argv[11]) >> boolalpha >> skip_pruning;
   istringstream(argv[12]) >> boolalpha >> dfs;




/**
 * 2. read the directed graph
 */

    Io oGraph;
    oGraph.readFromFile(/*argv[2]*/input_file);
    printf("Graph info: n = %d, m = %d\n", oGraph.m_uiN, oGraph.m_uiM);
    printf("Query parameters: insertion = %d, deletion = %d\n", num_insertion_edges, num_deletion_edges);
    vector<pair<uint32_t,uint32_t>> input_graph;   //store the original edges in the input graph
    oGraph.getEdges(input_graph);

/**
 * 3. generate edges to be inserted-deleted / or load the generated edges
 */
    edgeGenerator m_generator(input_graph);
    if(generate_subgraphs){
        string output_subfile_string= file_path + file_name + "/" + file_name;
        //char *output_file = (char*)output_subfile_string.c_str();
        m_generator.generatorSubGraphs(input_graph, output_subfile_string);
        return 0;
    }
    if (generate_modefiy_edges){   //true: generate edges to be inserted-deleted / false: read the generated edges
        m_generator.generatorInsertEdges(num_insertion_edges, generated_insert_file);
        m_generator.generatorDeleteEdges(num_deletion_edges, generated_delete_file);
        return 0;
    }
    std::vector<std::pair<uint32_t, uint32_t>> edges_to_be_inserted, edges_to_be_deleted;
    oGraph.readModifiedEdges(edges_to_be_inserted, generated_insert_file);
    oGraph.readModifiedEdges(edges_to_be_deleted, generated_delete_file);

    //obtain the bi-directional graph
    //m_generator.getBiGraph(bi_graph_file);

    printf("Parameter settings: insertion edge file = %zu, deletion edge file = %zu, batch size = %d\n", edges_to_be_inserted.size(), edges_to_be_deleted.size(), batch_size);

/**
 * 4. do the initial decomposition
 */

    vector<vector<pair<::uint32_t,::uint32_t>>> independent_k_lists, d_core_decomposition;
    ifstream exist;
    exist.open(decomposition_result_file);
    if(exist){   //read exsiting d-core decomposition result from file
        oGraph.readFromFile(decomposition_result_file, d_core_decomposition);
        printf("Read existing decomposition result from file\n");
    }
    else{
        const auto beg = std::chrono::steady_clock::now();
        Repeel oRepeel(input_graph);
        const auto end1 = std::chrono::steady_clock::now();
        oRepeel.peelKlist(independent_k_lists);
        const auto end2 = std::chrono::steady_clock::now();
        oRepeel.peelDcore(d_core_decomposition);
        const auto end3 = std::chrono::steady_clock::now();
        const auto end = std::chrono::steady_clock::now();
        printf("Initial decomposition total costs \x1b[1;31m%f\x1b[0m ms, in costs \x1b[1;31m%f\x1b[0m ms, out costs \x1b[1;31m%f\x1b[0m ms\n",
               std::chrono::duration<double, std::milli>(end - beg).count(),
               std::chrono::duration<double, std::milli>(end2 - end1).count(),
               std::chrono::duration<double, std::milli>(end3 - end2).count());
   }




/**
 * 5. do the naive re-peel maintenance
 */
    if(decomp){
        const auto beg1 = omp_get_wtime();
        //insertion
        edgeGenerator m_generator_insert(input_graph);
        m_generator_insert.m_insert_edges.clear();
        for(uint32_t  j = 0; j < batch_size; ++j){
            m_generator_insert.m_insert_edges.push_back(edges_to_be_inserted[j]);
        }
        const auto test0 = omp_get_wtime();
        std::vector<std::pair<uint32_t, uint32_t>> insert_modified_graph = m_generator_insert.getInsertedGraph();
        const auto test00 = omp_get_wtime();
        Repeel iRepeel(insert_modified_graph);
        vector<vector<pair<::uint32_t,::uint32_t>>> insert_independent_k_lists, insert_d_core_decomposition;
        iRepeel.peelKlist(insert_independent_k_lists);
        iRepeel.peelDcore(insert_d_core_decomposition);
        const auto end1_1 = omp_get_wtime();

        //deletion
        const auto test1 = omp_get_wtime();
        edgeGenerator m_generator_delete(input_graph);
        m_generator_delete.m_delete_edges.clear();
        for(uint32_t  j = 0 ; j < batch_size; ++j){
            m_generator_delete.m_delete_edges.push_back(edges_to_be_deleted[j]);
        }
        const auto test2 = omp_get_wtime();
        std::vector<std::pair<uint32_t, uint32_t>> deleted_modified_graph = m_generator_delete.getDeletedGraph();
        const auto test3 = omp_get_wtime();
        Repeel dRepeel(deleted_modified_graph);
        vector<vector<pair<::uint32_t,::uint32_t>>> delete_independent_k_lists, delete_d_core_decomposition;
        dRepeel.peelKlist(delete_independent_k_lists);
        dRepeel.peelDcore(delete_d_core_decomposition);
        const auto test4 = omp_get_wtime();

        const auto end1_2 = omp_get_wtime();
        const auto dif1_1 = end1_1 - beg1;
        const auto dif1_2 = end1_2 - end1_1;


        printf("Repeel insertion costs \x1b[1;31m%f\x1b[0m ms; deletion costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms\n",
           dif1_1*1000,//std::chrono::duration<double, std::milli>(dif1_1).count(),
           ((test2 - test1 )+ (test4 - test3) + (test00 - test0))*1000,
           ((test2 - test1 )+ (test4 - test3)+ (test00 - test0) + dif1_1)*1000);//std::chrono::duration<double, std::milli>(dif1_2).count(),
           //std::chrono::duration<double, std::milli>(dif1_1 + dif1_2).count());

        
    }


/**
 * 6. do the optimized re-peel maintenance (2019 TKDE paper)
 */
    if(peel){
        const auto beg2 = omp_get_wtime();
        edgeGenerator m_generator_optimized_insert(input_graph);
        vector<vector<pair<uint32_t, uint32_t>>>  new_d_core_decomposition;

        //insertion
        m_generator_optimized_insert.m_insert_edges.clear();
        for(uint32_t  j = 0; j < batch_size; j++){
            m_generator_optimized_insert.m_insert_edges.push_back(edges_to_be_inserted[j]);
        }
        const auto test0 = omp_get_wtime();
        std::vector<std::pair<uint32_t, uint32_t>> insert_modified_graph_optimized = m_generator_optimized_insert.getInsertedGraph();
        const auto test00 = omp_get_wtime();
        Repeel iRepeel_optimized(insert_modified_graph_optimized);
        iRepeel_optimized.optimizedPeelDcore(m_generator_optimized_insert.m_insert_edges,
                                             m_generator_optimized_insert.new_to_old_node_map,
                                             true,
                                             new_d_core_decomposition, d_core_decomposition);
        const auto end2_1 = omp_get_wtime();

        //deletion
        const auto test1 = omp_get_wtime();
        new_d_core_decomposition.clear();
        edgeGenerator m_generator_optimized_delete(input_graph);
        m_generator_optimized_delete.m_delete_edges.clear();
        for(uint32_t  j = 0; j < batch_size; j++){
            m_generator_optimized_delete.m_delete_edges.push_back(edges_to_be_deleted[j]);
        }
        const auto test2 = omp_get_wtime();
        std::vector<std::pair<uint32_t, uint32_t>> delete_modified_graph_optimized = m_generator_optimized_delete.getDeletedGraph();
        const auto test3 = omp_get_wtime();
        Repeel dRepeel_optimized(delete_modified_graph_optimized);
        dRepeel_optimized.optimizedPeelDcore(m_generator_optimized_delete.m_delete_edges,
                                             m_generator_optimized_delete.new_to_old_node_map,
                                             false,
                                             new_d_core_decomposition, d_core_decomposition);
        const auto test4 = omp_get_wtime();

        const auto end2_2 = omp_get_wtime();
        const auto dif2_1 = end2_1 - beg2;
        const auto dif2_2 = end2_2 - end2_1;


        printf("Optimized repeel insertion costs \x1b[1;31m%f\x1b[0m ms; deletion costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms \n",
           dif2_1*1000,//std::chrono::duration<double, std::milli>(dif2_1).count(),
           ((test2 - test1 )+ (test4 - test3) + (test00 - test0))*1000,
           ((test2 - test1 )+ (test4 - test3)+ (test00 - test0) + dif2_1)*1000);//std::chrono::duration<double, std::milli>(dif2_1 + dif2_2).count());


    }


/**
 * 7. do the DFS-based maintenance
 */
    const auto beg3 = std::chrono::steady_clock::now();

    edgeGenerator m_generator_dfs_insert(input_graph);
    edgeGenerator m_generator_dfs_delete(input_graph);


    /**
     * 7.1. do the DFS-based maintenance without parallelization (processing one edge at a time)
     */
    if(dfs){
        /*edge insertion*/
        /*std::chrono::duration<double>*/double insert_kmax_time = 0, insert_lmax_time = 0, delete_kmax_time = 0, delete_lmax_time = 0;
        DfsSearch dfs_insert(input_graph, d_core_decomposition);
        for (::uint32_t i = 0; i < batch_size; ++i) {   //process edges one by one
            //const auto test0 = std::chrono::steady_clock::now();
            auto test0 = omp_get_wtime();
            dfs_insert.insertEdge(edges_to_be_inserted[i]);
            vector<pair<uint32_t, uint32_t>> tmp_insert_edge_vec {edges_to_be_inserted[i]};
            //const auto test1 = std::chrono::steady_clock::now();
            auto test1 = omp_get_wtime();
            uint32_t M = min(dfs_insert.k_max[edges_to_be_inserted[i].first], dfs_insert.k_max[edges_to_be_inserted[i].second]);
            dfs_insert.maintainKmax(tmp_insert_edge_vec, true, M);
            //const auto test2 = std::chrono::steady_clock::now();
            auto test2 = omp_get_wtime();
            dfs_insert.maintainKlist(tmp_insert_edge_vec, true, M, k0core_pruning,use_h_index, lmax_number_of_threads, reuse_pruning, skip_pruning);
            //const auto test3 = std::chrono::steady_clock::now();
            auto test3 = omp_get_wtime();
            insert_kmax_time += test2 - test1;
            insert_lmax_time += test3 - test2;
//            printf("DFS-based insertion in costs \x1b[1;31m%f\x1b[0m ms; out costs \x1b[1;31m%f\x1b[0m ms; preperation costs \x1b[1;31m%f\x1b[0m ms\n",
//                   std::chrono::duration<double, std::milli>(test2 - test1).count(),
//                   std::chrono::duration<double, std::milli>(test3 - test2).count(),
//                   std::chrono::duration<double, std::milli>(test1 - test0).count());
        }

        /*edge deletion*/
        const auto end3_1 = std::chrono::steady_clock::now();
        DfsSearch dfs_delete(input_graph, d_core_decomposition);
        for(::uint32_t i = 0 ; i <  batch_size; ++i){ //process edges one by one
            //const auto test0 = std::chrono::steady_clock::now();
            auto test0 = omp_get_wtime();
            dfs_delete.deleteEdge(edges_to_be_deleted[i]);
            vector<pair<uint32_t, uint32_t>> tmp_delete_edge_vec {edges_to_be_deleted[i]};
            //vector<vector<pair<::uint32_t,::uint32_t>>> dfs_delete_independent_k_lists, dfs_delete_d_core_decomposition;
            //const auto test1 = std::chrono::steady_clock::now();
            auto test1 = omp_get_wtime();
            uint32_t M = min(dfs_delete.k_max[edges_to_be_deleted[i].first], dfs_delete.k_max[edges_to_be_deleted[i].second]);
            //uint32_t N = max(dfs_delete.k_max[edges_to_be_deleted[i].first], dfs_delete.k_max[edges_to_be_deleted[i].second]);
            dfs_delete.maintainKmax(tmp_delete_edge_vec, false, M);
            //const auto test2 = std::chrono::steady_clock::now();
            auto test2 = omp_get_wtime();
            dfs_delete.maintainKlist(tmp_delete_edge_vec, false, M, k0core_pruning,use_h_index,lmax_number_of_threads,reuse_pruning,skip_pruning);
            //const auto test3 = std::chrono::steady_clock::now();
            auto test3 = omp_get_wtime();
            delete_kmax_time += test2 - test1;
            delete_lmax_time += test3 - test2;
//            printf("DFS-based deletion in costs \x1b[1;31m%f\x1b[0m ms; out costs \x1b[1;31m%f\x1b[0m ms; preperation costs \x1b[1;31m%f\x1b[0m ms, %d \n",
//                   std::chrono::duration<double, std::milli>(test2 - test1).count(),
//                   std::chrono::duration<double, std::milli>(test3 - test2).count(),
//                   std::chrono::duration<double, std::milli>(test1 - test0).count(), i);
        }

        const auto end3_2 = std::chrono::steady_clock::now();
        const auto dif3_1 = end3_1 - beg3;
        const auto dif3_2 = end3_2 - end3_1;
        // printf("DFS-based insertion costs \x1b[1;31m%f\x1b[0m ms; deletion costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms\n",
        //        std::chrono::duration<double, std::milli>(dif3_1).count(),
        //        std::chrono::duration<double, std::milli>(dif3_2).count(),
        //        std::chrono::duration<double, std::milli>(dif3_1 + dif3_2).count());
        printf("DFS-based insertion kmax costs \x1b[1;31m%f\x1b[0m ms; insertion lmax costs \x1b[1;31m%f\x1b[0m ms; insertion total costs \x1b[1;31m%f\x1b[0m ms;\n deletion kmax costs \x1b[1;31m%f\x1b[0m ms; deletion lmax costs \x1b[1;31m%f\x1b[0m ms; deletion total costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms;\n",
               /*std::chrono::duration<double, std::milli>(insert_kmax_time).count()*/insert_kmax_time*1000,
               /*std::chrono::duration<double, std::milli>(insert_lmax_time).count()*/insert_lmax_time*1000,
               (insert_kmax_time + insert_lmax_time)*1000,
               /*std::chrono::duration<double, std::milli>(delete_kmax_time).count()*/delete_kmax_time*1000,
               /*std::chrono::duration<double, std::milli>(delete_lmax_time).count()*/delete_lmax_time*1000,
               (delete_kmax_time + delete_lmax_time)*1000,
               /*std::chrono::duration<double, std::milli>(insert_kmax_time + insert_lmax_time + delete_kmax_time + delete_lmax_time).count()*/(insert_kmax_time + insert_lmax_time + delete_kmax_time + delete_lmax_time)*1000);
    }
    /**
     * 7.2. do the DFS-based maintenance with parallelization (processing batch by batch, each batch contains multiple edges)
     */
//    else{  //only do initialization once for all edge batches, process lmax in parallel using the DFS-based single-edge lmax maintenance algorithm, parallelize different k-lists
//        /*edge insertion with dfs*/
//        /*generate edge batches*/
//        const auto test1 = std::chrono::steady_clock::now();
//
//        vector<vector<pair<uint32_t, uint32_t>>> insert_edge_batch_parallel;
//        vector<pair<uint32_t, uint32_t>> remaining_unbatched_edges;
//        m_generator_dfs_insert.m_insert_edges.clear();
//        for(uint32_t j = 0; j < batch_size; j++){
//            m_generator_dfs_insert.m_insert_edges.push_back(edges_to_be_inserted[j]);
//        }
//        insert_edge_batch_parallel = edgeGenerator::getEdgeBatch(m_generator_dfs_insert.m_insert_edges, d_core_decomposition, remaining_unbatched_edges);
//        DfsSearch dfs_insert_kmax(input_graph, d_core_decomposition), dfs_insert_lmax(input_graph, d_core_decomposition);
//
//        const auto test2 = std::chrono::steady_clock::now();
//
//
//        //insertion: parallel batch k_max maintenance
//        uint32_t max_M_value_insert = 0;
//        for(const auto & batch : insert_edge_batch_parallel){
//            for(const auto & edge : batch){
//                dfs_insert_kmax.insertEdge(edge);
//            }
//            #pragma omp parallel for num_threads(THE_NUMBER_OF_THREADS)
//            for (const auto & edge : batch) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                //dfs_insert.insertEdge(edge);
//                const auto test1 = std::chrono::steady_clock::now();
//                uint32_t M = min(dfs_insert_kmax.k_max[edge.first], dfs_insert_kmax.k_max[edge.second]);
//                max_M_value_insert = max(max_M_value_insert, M);
//                dfs_insert_kmax.maintainKmax(tmp_edge_vec, true, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_insert.maintainKlist(tmp_edge_vec, true, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "kmax parallel insert done" << endl;
//        const auto test3 = std::chrono::steady_clock::now();
//
//        //insertion: sequential k_max maintenance for unbathced edges
//        if(!remaining_unbatched_edges.empty()){
//            //DfsSearch dfs_insert_unbatched(input_graph, d_core_decomposition);
//            for (const auto & edge : remaining_unbatched_edges) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                dfs_insert_kmax.insertEdge(edge);
//                uint32_t M = min(dfs_insert_kmax.k_max[edge.first], dfs_insert_kmax.k_max[edge.second]);
//                max_M_value_insert = max(max_M_value_insert, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_insert_kmax.maintainKmax(tmp_edge_vec, true, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_insert_unbatched.maintainKlist(tmp_edge_vec, true, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "kmax sequential insert done" << endl;
//        const auto test4 = std::chrono::steady_clock::now();
//
//        //insertion: synchronize k_max and l_max
//        for(uint32_t vid = 0; vid < dfs_insert_kmax.k_max.size(); ++vid){
//            dfs_insert_lmax.k_max[vid] = dfs_insert_kmax.k_max[vid];
//            dfs_insert_lmax.l_max[vid] = dfs_insert_kmax.l_max[vid];
//        }
//
//        //insertion: parallel l_max maintenance
//        for(uint32_t j = 0; j < batch_size; j++){
//            auto edge = edges_to_be_inserted[j];
//            vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//            dfs_insert_lmax.insertEdge(edge);
//            uint32_t M = min(dfs_insert_lmax.k_max[edge.first], dfs_insert_lmax.k_max[edge.second]);
//            dfs_insert_lmax.maintainKlist(tmp_edge_vec, true, M, k0core_pruning);
//        }
//        const auto test5 = std::chrono::steady_clock::now();
//
//        printf("DFS-based insertion preperation costs \x1b[1;31m%f\x1b[0m ms; parallel kmax costs \x1b[1;31m%f\x1b[0m ms; sequential kmax costs \x1b[1;31m%f\x1b[0m ms; lmax costs \x1b[1;31m%f\x1b[0m ms\n",
//               std::chrono::duration<double, std::milli>(test2 - test1).count(),
//               std::chrono::duration<double, std::milli>(test3 - test2).count(),
//               std::chrono::duration<double, std::milli>(test4 - test3).count(),
//               std::chrono::duration<double, std::milli>(test5 - test4).count());
//
//
//        const auto end3_1 = std::chrono::steady_clock::now();
//
//
//        ////============================================================================================================
//        /*edge deletion with dfs*/
//
//        /*generate edge batches*/
//        vector<vector<pair<uint32_t, uint32_t>>> delete_edge_batch_parallel;
//        m_generator_dfs_delete.m_delete_edges.clear();
//        for(uint32_t j = 0; j < batch_size; ++j){
//            m_generator_dfs_delete.m_delete_edges.push_back(edges_to_be_deleted[j]);
//        }
//        delete_edge_batch_parallel = edgeGenerator::getEdgeBatch( m_generator_dfs_delete.m_delete_edges, d_core_decomposition, remaining_unbatched_edges);
//        DfsSearch dfs_delete_kmax(input_graph, d_core_decomposition), dfs_delete_lmax(input_graph, d_core_decomposition);
//
//        //deletion: parallel batch k_max maintenance
//        uint32_t max_M_value_delete = 0;
//        for(const auto& batch : delete_edge_batch_parallel){
//            for(const auto & edge : batch){
//                dfs_delete_kmax.deleteEdge(edge);
//            }
//            #pragma omp parallel for num_threads(THE_NUMBER_OF_THREADS)
//            for (const auto  &edge : batch) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                //dfs_delete.deleteEdge(edge);
//                uint32_t M = min(dfs_delete_kmax.k_max[edge.first], dfs_delete_kmax.k_max[edge.second]);
//                max_M_value_delete = max(max_M_value_delete, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_delete_kmax.maintainKmax(tmp_edge_vec, false, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_delete.maintainKlist(tmp_edge_vec, false, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "parallel deletion done" << endl;
//
//
//        //deletion: sequential k_max maintenance for unbathced edges
//        if(!remaining_unbatched_edges.empty()){
//            //DfsSearch dfs_delete_unbatched(input_graph, d_core_decomposition);
//            for (const auto & edge : remaining_unbatched_edges) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                dfs_delete_kmax.deleteEdge(edge);
//                uint32_t M = min(dfs_delete_kmax.k_max[edge.first], dfs_delete_kmax.k_max[edge.second]);
//                max_M_value_delete = max(max_M_value_delete, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_delete_kmax.maintainKmax(tmp_edge_vec, false, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_delete.maintainKlist(tmp_edge_vec, false, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "sequential deletion done" << endl;
//
//        //deletion: synchronize k_max and l_max
//        for(uint32_t vid = 0; vid < dfs_delete_kmax.k_max.size(); ++vid){
//            dfs_delete_lmax.k_max[vid] = dfs_delete_kmax.k_max[vid];
//            dfs_delete_lmax.l_max[vid] = dfs_delete_kmax.l_max[vid];
//        }
//
//        //deletion: parallel l_max maintenance
//        for(uint32_t j = 0; j < batch_size; j++){
//            auto edge = edges_to_be_deleted[j];
//            vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//            dfs_delete_lmax.deleteEdge(edge);
//            uint32_t M = min(dfs_delete_lmax.k_max[edge.first], dfs_delete_lmax.k_max[edge.second]);
//            dfs_delete_lmax.maintainKlist(tmp_edge_vec, false, M, k0core_pruning);
//        }
//
//
//        const auto end3_2 = std::chrono::steady_clock::now();
//        const auto dif3_1 = end3_1 - beg3;
//        const auto dif3_2 = end3_2 - end3_1;
//        printf("DFS-based insertion costs \x1b[1;31m%f\x1b[0m ms; deletion costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms\n",
//               std::chrono::duration<double, std::milli>(dif3_1).count(),
//               std::chrono::duration<double, std::milli>(dif3_2).count(),
//               std::chrono::duration<double, std::milli>(dif3_1 + dif3_2).count());
//    }

////============================================================================================================

//    else{  //only do initialization once for all edge batches, process lmax in parallel by inserting edges in parallel, parallelize different edges
//        /*edge insertion with dfs*/
//        /*generate edge batches*/
//        const auto out_test1 = std::chrono::steady_clock::now();
//
//        vector<vector<pair<uint32_t, uint32_t>>> insert_edge_batch_parallel;
//        vector<pair<uint32_t, uint32_t>> remaining_unbatched_edges;
//        m_generator_dfs_insert.m_insert_edges.clear();
//        for(uint32_t j = 0; j < batch_size; j++){
//            m_generator_dfs_insert.m_insert_edges.push_back(edges_to_be_inserted[j]);
//        }
//        insert_edge_batch_parallel = edgeGenerator::getEdgeBatch(m_generator_dfs_insert.m_insert_edges, d_core_decomposition, remaining_unbatched_edges,kmax_hierarchy, kedge_set);
//        DfsSearch dfs_insert_kmax(input_graph, d_core_decomposition), dfs_insert_lmax(input_graph, d_core_decomposition);
//
//        const auto out_test2 = std::chrono::steady_clock::now();
//
//
//        //insertion: parallel batch k_max maintenance
//        uint32_t max_M_value_insert = 0;
//        for(const auto & batch : insert_edge_batch_parallel){
//            for(const auto & edge : batch){
//                dfs_insert_kmax.insertEdge(edge);
//            }
//            #pragma omp parallel for num_threads(the_number_of_threads)
//            for (const auto & edge : batch) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                const auto test1 = std::chrono::steady_clock::now();
//                uint32_t M = min(dfs_insert_kmax.k_max[edge.first], dfs_insert_kmax.k_max[edge.second]);
//                max_M_value_insert = max(max_M_value_insert, M);
//                dfs_insert_kmax.maintainKmax(tmp_edge_vec, true, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_insert.maintainKlist(tmp_edge_vec, true, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "kmax parallel insert done" << endl;
//        const auto out_test3 = std::chrono::steady_clock::now();
//
//
//        //insertion: sequential k_max maintenance for unbathced edges
//        if(!remaining_unbatched_edges.empty()){
//            for (const auto & edge : remaining_unbatched_edges) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                dfs_insert_kmax.insertEdge(edge);
//                uint32_t M = min(dfs_insert_kmax.k_max[edge.first], dfs_insert_kmax.k_max[edge.second]);
//                max_M_value_insert = max(max_M_value_insert, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_insert_kmax.maintainKmax(tmp_edge_vec, true, M);
//                const auto test2 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "kmax sequential insert done" << endl;
//        const auto out_test4 = std::chrono::steady_clock::now();
//
//        //insertion: synchronize k_max and l_max
//        for(uint32_t vid = 0; vid < dfs_insert_kmax.k_max.size(); ++vid){
//            dfs_insert_lmax.k_max[vid] = dfs_insert_kmax.k_max[vid];
//            dfs_insert_lmax.l_max[vid] = dfs_insert_kmax.l_max[vid];
//        }
//
//        //insertion: parallel l_max maintenance
//        for(uint32_t j = 0; j < batch_size; j++){
//            auto edge = edges_to_be_inserted[j];
//            vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//            dfs_insert_lmax.insertEdge(edge);
//            uint32_t M = min(dfs_insert_lmax.k_max[edge.first], dfs_insert_lmax.k_max[edge.second]);
//            dfs_insert_lmax.maintainKlist(tmp_edge_vec, true, M, k0core_pruning,use_h_index,lmax_number_of_threads);
//        }
//        const auto out_test5 = std::chrono::steady_clock::now();
//
//
//        printf("DFS-based insertion preperation costs \x1b[1;31m%f\x1b[0m ms; parallel kmax costs \x1b[1;31m%f\x1b[0m ms; sequential kmax costs \x1b[1;31m%f\x1b[0m ms; lmax costs \x1b[1;31m%f\x1b[0m ms\n",
//               std::chrono::duration<double, std::milli>(out_test2 - out_test1).count(),
//               std::chrono::duration<double, std::milli>(out_test3 - out_test2).count(),
//               std::chrono::duration<double, std::milli>(out_test4 - out_test3).count(),
//               std::chrono::duration<double, std::milli>(out_test5 - out_test4).count());
//
//
//        const auto end3_1 = std::chrono::steady_clock::now();
//
//
//        ////============================================================================================================
//        /*edge deletion with dfs*/
//
//        /*generate edge batches*/
//        vector<vector<pair<uint32_t, uint32_t>>> delete_edge_batch_parallel;
//        m_generator_dfs_delete.m_delete_edges.clear();
//        for(uint32_t j = 0; j < batch_size; ++j){
//            m_generator_dfs_delete.m_delete_edges.push_back(edges_to_be_deleted[j]);
//        }
//        delete_edge_batch_parallel = edgeGenerator::getEdgeBatch( m_generator_dfs_delete.m_delete_edges, d_core_decomposition, remaining_unbatched_edges,kmax_hierarchy, kedge_set);
//        DfsSearch dfs_delete_kmax(input_graph, d_core_decomposition), dfs_delete_lmax(input_graph, d_core_decomposition);
//
//        //deletion: parallel batch k_max maintenance
//        const auto out_test6 = std::chrono::steady_clock::now();
//        uint32_t max_M_value_delete = 0;
//        for(const auto& batch : delete_edge_batch_parallel){
//            for(const auto & edge : batch){
//                dfs_delete_kmax.deleteEdge(edge);
//            }
//
//            #pragma omp parallel for num_threads(the_number_of_threads)
//            for(const auto &edge : batch){
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                //dfs_delete.deleteEdge(edge);
//                uint32_t M = min(dfs_delete_kmax.k_max[edge.first], dfs_delete_kmax.k_max[edge.second]);
//                max_M_value_delete = max(max_M_value_delete, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_delete_kmax.maintainKmax(tmp_edge_vec, false, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_delete.maintainKlist(tmp_edge_vec, false, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "parallel deletion done" << endl;
//        const auto out_test7 = std::chrono::steady_clock::now();
//
//        //deletion: sequential k_max maintenance for unbathced edges
//        if(!remaining_unbatched_edges.empty()){
//            //DfsSearch dfs_delete_unbatched(input_graph, d_core_decomposition);
//            for (const auto & edge : remaining_unbatched_edges) {   //process edges one by one
//                const auto test0 = std::chrono::steady_clock::now();
//                vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//                dfs_delete_kmax.deleteEdge(edge);
//                uint32_t M = min(dfs_delete_kmax.k_max[edge.first], dfs_delete_kmax.k_max[edge.second]);
//                max_M_value_delete = max(max_M_value_delete, M);
//                const auto test1 = std::chrono::steady_clock::now();
//                dfs_delete_kmax.maintainKmax(tmp_edge_vec, false, M);
//                const auto test2 = std::chrono::steady_clock::now();
//                //dfs_delete.maintainKlist(tmp_edge_vec, false, M);
//                const auto test3 = std::chrono::steady_clock::now();
//            }
//        }
//        cout << "sequential deletion done" << endl;
//        const auto out_test8 = std::chrono::steady_clock::now();
//
//        //deletion: synchronize k_max and l_max
//        for(uint32_t vid = 0; vid < dfs_delete_kmax.k_max.size(); ++vid){
//            dfs_delete_lmax.k_max[vid] = dfs_delete_kmax.k_max[vid];
//            dfs_delete_lmax.l_max[vid] = dfs_delete_kmax.l_max[vid];
//        }
//        //deletion: parallel l_max maintenance
//        for(uint32_t j = 0; j < batch_size; j++){
//            auto edge = edges_to_be_deleted[j];
//            vector<pair<uint32_t, uint32_t>> tmp_edge_vec {edge};
//            dfs_delete_lmax.deleteEdge(edge);
//            uint32_t M = min(dfs_delete_lmax.k_max[edge.first], dfs_delete_lmax.k_max[edge.second]);
//            dfs_delete_lmax.maintainKlist(tmp_edge_vec, false, M, k0core_pruning,use_h_index,lmax_number_of_threads);
//        }
//        const auto out_test9 = std::chrono::steady_clock::now();
//
//        const auto end3_2 = std::chrono::steady_clock::now();
//        const auto dif3_1 = end3_1 - beg3;
//        const auto dif3_2 = end3_2 - end3_1;
//        printf("DFS-based insertion costs \x1b[1;31m%f\x1b[0m ms; deletion costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms\n",
//               std::chrono::duration<double, std::milli>(dif3_1).count(),
//               std::chrono::duration<double, std::milli>(dif3_2).count(),
//               std::chrono::duration<double, std::milli>(dif3_1 + dif3_2).count());
//    }

////============================================================================================================

    if(kmax_hierarchy || kedge_set) {
        //process kmax and lmax in parallel by processing batch of edges in parallel with H-index based method, parallelize different edges

        /*edge insertion with H-index*/
        double insert_kmax_time = 0, insert_lmax_time = 0, delete_kmax_time = 0, delete_lmax_time = 0;
        hIndex hindex_insert(input_graph, d_core_decomposition);

        /*generate edge batches*/
        const auto out_test1 = omp_get_wtime();
        vector<vector<pair<uint32_t, uint32_t>>> insert_edge_batch_parallel;
        vector<pair<uint32_t, uint32_t>> remaining_unbatched_edges, total_inserted_edge_batch;
        m_generator_dfs_insert.m_insert_edges.clear();
        for(uint32_t j = 0; j < batch_size; j++){
            m_generator_dfs_insert.m_insert_edges.push_back(edges_to_be_inserted[j]);
            total_inserted_edge_batch.push_back(edges_to_be_inserted[j]);
        }
        insert_edge_batch_parallel = edgeGenerator::getEdgeBatch(m_generator_dfs_insert.m_insert_edges, d_core_decomposition, remaining_unbatched_edges,kmax_hierarchy, kedge_set);
        const auto out_test2 = omp_get_wtime();

        printf("insertion batch size: %d, construct costs \x1b[1;31m%f\x1b[0m ms\n", insert_edge_batch_parallel.size(),
               /*std::chrono::duration<double, std::milli>(out_test2 - out_test1).count()*/(out_test2 - out_test1)*1000);

        //insertion: parallel batch k_max maintenance
        uint32_t max_M_value_insert = 0;
        for(const auto & batch : insert_edge_batch_parallel){
            for(const auto & edge : batch){
                hindex_insert.insertEdge(edge);
                uint32_t M = min(hindex_insert.k_max[edge.first], hindex_insert.k_max[edge.second]);
                max_M_value_insert = max(max_M_value_insert, M);
            }
            /*perform the H-index-based refinement*/
            const auto test1 = omp_get_wtime();
            hindex_insert.maintainKmax(batch, true, lmax_number_of_threads);
            const auto test2 = omp_get_wtime();
            insert_kmax_time += test2 - test1;
        }
        const auto out_test3 = omp_get_wtime();


        //insertion: sequential k_max maintenance for unbathced edges
        if(!remaining_unbatched_edges.empty()){
            const auto test1 = omp_get_wtime();
            vector<pair<uint32_t, uint32_t>> tmp_edge_vec;
            for (const auto & edge : remaining_unbatched_edges) {
                tmp_edge_vec.push_back(edge);
                uint32_t M = min(hindex_insert.k_max[edge.first], hindex_insert.k_max[edge.second]);
                max_M_value_insert = max(max_M_value_insert, M);
            }
            hindex_insert.maintainKmaxSingle(tmp_edge_vec, true, lmax_number_of_threads);
            const auto test2 = omp_get_wtime();
            insert_kmax_time += test2 - test1;
        }
        const auto out_test4 = omp_get_wtime();


        //insertion: parallel l_max maintenance
        hindex_insert.maintainKlist(total_inserted_edge_batch, true,
                                        max_M_value_insert, k0core_pruning, reuse_pruning, lmax_number_of_threads);

        const auto out_test5 = omp_get_wtime();
        insert_lmax_time += out_test5 - out_test4;





        /*edge deletion with h-index based method*/
        hIndex hindex_delete(input_graph, d_core_decomposition);
        /*generate edge batches*/
        const auto delete_batch_start = omp_get_wtime();
        vector<vector<pair<uint32_t, uint32_t>>> delete_edge_batch_parallel;
        m_generator_dfs_delete.m_delete_edges.clear();
        for(uint32_t j = 0; j < batch_size; j++){
            m_generator_dfs_delete.m_delete_edges.push_back(edges_to_be_deleted[j]);
        }
        delete_edge_batch_parallel = edgeGenerator::getEdgeBatch(m_generator_dfs_delete.m_delete_edges, d_core_decomposition, remaining_unbatched_edges,kmax_hierarchy, kedge_set);
        const auto out_test6 = omp_get_wtime();

        printf("deletion batch size: %d, construct costs \x1b[1;31m%f\x1b[0m ms\n", delete_edge_batch_parallel.size(),
               (out_test6 - delete_batch_start)*1000);

        //deletion: parallel batch k_max maintenance
        uint32_t max_M_value_delete = 0;
        for(const auto & batch : delete_edge_batch_parallel){
            for(const auto & edge : batch){
                hindex_delete.deleteEdge(edge);
                uint32_t M = min(hindex_delete.k_max[edge.first], hindex_delete.k_max[edge.second]);
                max_M_value_delete = max(max_M_value_delete, M);
            }
            /*perform the H-index-based refinement*/
            const auto test1 = omp_get_wtime();
            hindex_delete.maintainKmax(batch, false, lmax_number_of_threads);
            const auto test2 = omp_get_wtime();
            delete_kmax_time += test2 - test1;
        }
        const auto out_test7 = omp_get_wtime();

        //deletion: sequential k_max maintenance for unbathced edges
        if(!remaining_unbatched_edges.empty()){
            const auto test1 = omp_get_wtime();
            vector<pair<uint32_t, uint32_t>> tmp_edge_vec;
            for (const auto & edge : remaining_unbatched_edges) {
                tmp_edge_vec.push_back(edge);
                uint32_t M = min(hindex_delete.k_max[edge.first], hindex_delete.k_max[edge.second]);
                max_M_value_delete = max(max_M_value_delete, M);
            }
            hindex_delete.maintainKmaxSingle(tmp_edge_vec, false, lmax_number_of_threads);
            const auto test2 = omp_get_wtime();
            delete_kmax_time += test2 - test1;
        }
        const auto out_test8 = omp_get_wtime();

        //deletion: parallel l_max maintenance
        hindex_delete.maintainKlist(total_inserted_edge_batch, false,
                                        max_M_value_delete, k0core_pruning, reuse_pruning, lmax_number_of_threads);
        const auto out_test9 = omp_get_wtime();
        delete_lmax_time += out_test9 - out_test8;



        printf("Hindex-based insertion kmax costs \x1b[1;31m%f\x1b[0m ms; insertion lmax costs \x1b[1;31m%f\x1b[0m ms; insertion total costs \x1b[1;31m%f\x1b[0m ms;\n deletion kmax costs \x1b[1;31m%f\x1b[0m ms; deletion lmax costs \x1b[1;31m%f\x1b[0m ms; deletion total costs \x1b[1;31m%f\x1b[0m ms; total costs \x1b[1;31m%f\x1b[0m ms;\n",
               (insert_kmax_time)*1000,
               (insert_lmax_time)*1000,
               (insert_kmax_time + insert_lmax_time)*1000,
               (delete_kmax_time)*1000,
               (delete_lmax_time)*1000,
               (delete_kmax_time + delete_lmax_time)*1000,
               (insert_kmax_time + insert_lmax_time + delete_kmax_time + delete_lmax_time)*1000);
    }

    return 0;

    /**
     * h-index refinement based method for single-edge updates, not used in the paper because of bad performance
     *
     *      hIndex hindex_insert(input_graph, d_core_decomposition);
            for (::uint32_t i = 0; i < batch_size; ++i) {   //process edges one by one
                const auto test0 = std::chrono::steady_clock::now();
                hindex_insert.insertEdge(edges_to_be_inserted[i]);
                vector<pair<uint32_t, uint32_t>> tmp_insert_edge_vec {edges_to_be_inserted[i]};
                const auto test1 = std::chrono::steady_clock::now();
                uint32_t M = min(hindex_insert.k_max[edges_to_be_inserted[i].first], hindex_insert.k_max[edges_to_be_inserted[i].second]);
                hindex_insert.maintainKmax(tmp_insert_edge_vec, true, M, lmax_number_of_threads);
                const auto test2 = std::chrono::steady_clock::now();
                hindex_insert.maintainKlist(tmp_insert_edge_vec, true, M, k0core_pruning, lmax_number_of_threads);
                const auto test3 = std::chrono::steady_clock::now();
                insert_kmax_time += test2 - test1;
                insert_lmax_time += test3 - test2;
//            printf("DFS-based insertion in costs \x1b[1;31m%f\x1b[0m ms; out costs \x1b[1;31m%f\x1b[0m ms; preperation costs \x1b[1;31m%f\x1b[0m ms\n",
//                   std::chrono::duration<double, std::milli>(test2 - test1).count(),
//                   std::chrono::duration<double, std::milli>(test3 - test2).count(),
//                   std::chrono::duration<double, std::milli>(test1 - test0).count());
            }
            printf("insertion hindex done \n");
            oGraph.writeToFile("../optimized-hindex-insert.txt", hindex_insert.l_max);

            hIndex hindex_delete(input_graph, d_core_decomposition);
            for(::uint32_t i = 0 ; i <  batch_size; ++i){ //process edges one by one
                const auto test0 = std::chrono::steady_clock::now();
                hindex_delete.deleteEdge(edges_to_be_deleted[i]);
                vector<pair<uint32_t, uint32_t>> tmp_delete_edge_vec {edges_to_be_deleted[i]};
                const auto test1 = std::chrono::steady_clock::now();
                uint32_t M = min(hindex_delete.k_max[edges_to_be_deleted[i].first], hindex_delete.k_max[edges_to_be_deleted[i].second]);
                //uint32_t N = max(dfs_delete.k_max[edges_to_be_deleted[i].first], dfs_delete.k_max[edges_to_be_deleted[i].second]);
                hindex_delete.maintainKmax(tmp_delete_edge_vec, false, M, lmax_number_of_threads);
                printf("deletion kmax done \n");
                const auto test2 = std::chrono::steady_clock::now();
                hindex_delete.maintainKlist(tmp_delete_edge_vec, false, M, k0core_pruning,lmax_number_of_threads);
                printf("deletion lmax done \n");
                const auto test3 = std::chrono::steady_clock::now();
                delete_kmax_time += test2 - test1;
                delete_lmax_time += test3 - test2;
//            printf("DFS-based deletion in costs \x1b[1;31m%f\x1b[0m ms; out costs \x1b[1;31m%f\x1b[0m ms; preperation costs \x1b[1;31m%f\x1b[0m ms, %d \n",
//                   std::chrono::duration<double, std::milli>(test2 - test1).count(),
//                   std::chrono::duration<double, std::milli>(test3 - test2).count(),
//                   std::chrono::duration<double, std::milli>(test1 - test0).count(), i);
            }
            oGraph.writeToFile("../optimized-hindex-delete.txt", hindex_delete.l_max);
     */
}




/*
 *
 * this is the code for converting the Graph.dat file to the format edge list
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main()
{
    std::string filename = "message";
    std::ifstream fin("/Users/Exp-data/dcore-maintain/" + filename + "/Graph.dat");
    if (!fin.is_open()) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    int n, m = 0, cnt = 0;
    fin >> n;
    std::cout << n << " vertices\n";

    std::vector<std::vector<int>> E(n);
    std::vector<std::pair<int, int>> e;

    std::string line;
    std::getline(fin, line); // Read the first line
    for(int u = 0; u < n; u++){
        std::getline(fin, line);
        std::stringstream ss(line);
        int v;
        while(ss >> v){
            E[u].push_back(v);
            m++;
        }
    }
    fin.close();

    for(int u = 0; u < n; u++){
        for(int j = 1; j < E[u].size(); j++) {
            int v = E[u][j];
            e.push_back({u, v});
        }
    }

    std::ofstream fout("/Users/Exp-data/dcore-maintain/" + filename + "/" + filename + ".txt");
    if (!fout.is_open()) {
        std::cerr << "Failed to create file\n";
        return 1;
    }
    //fout << n << " " << m << std::endl;
    for(int i = 0; i < e.size(); i++){
        fout << e[i].first << " " << e[i].second << std::endl;
    }
    fout.close();

    return 0;
}

*/