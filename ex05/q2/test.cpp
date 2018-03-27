#include <stdio.h>
#include "markov_tree.hpp"
#include <cassert>
#include <map>
#include <functional>

#define IS_CLOSE(a, b) (abs((a) - (b)) < 1e-16)

void test_single() {
    const double val = 1.;
    node_t * root = init_node(val, NULL);
    assert(root->value == val);
    assert(root->num_children == 0);

    free_node(root);
}

void test_child(){
    node_t * root = init_node(1., NULL);
    node_t * c1 = init_node(2., root);
    assert(root->child == c1);
    assert(c1->parent == root);
    assert(c1->sibling_left == NULL);
    assert(c1->sibling_right == NULL);
    assert(root->num_children == 1);
    free_node(root);
}

void test_two_children(){
    node_t * root = init_node(1., NULL);
    node_t * c1 = init_node(2., root);
    node_t * c2 = init_node(3., root);

    assert(root->child == c2);
    assert(root->num_children == 2);

    assert(c1->parent == root);
    assert(c1->num_children == 0);
    assert(c2->parent == root);
    assert(c2->num_children == 0);


    assert(c1->sibling_left == c2);
    assert(c1->sibling_right == NULL);
    assert(c2->sibling_left == NULL);
    assert(c2->sibling_right == c1);

    free_node(root);
}

void test_three_children(){
    node_t * root = init_node(1., NULL);
    node_t * c1 = init_node(2., root);
    node_t * c2 = init_node(3., root);
    node_t * c3 = init_node(3., root);

    assert(root->child == c3);
    assert(c1->parent == root);
    assert(c2->parent == root);
    assert(c3->parent == root);

    assert(c1->sibling_left == c2);
    assert(c1->sibling_right == NULL);
    assert(c2->sibling_left == c3);
    assert(c2->sibling_right == c1);
    assert(c3->sibling_left == NULL);
    assert(c3->sibling_right == c2);

    free_node(root);
}

void test_nested_21(){
    node_t * root = init_node(1., NULL);
    node_t * c_r = init_node(2., root);
    node_t * c_l = init_node(3., root);

    node_t * c_l_r = init_node(4., c_l);
    node_t * c_l_l = init_node(5., c_l);

    assert(root->child == c_l);
    assert(c_l->parent == root);
    assert(c_r->parent == root);

    assert(c_l->sibling_left == NULL);
    assert(c_l->sibling_right == c_r);
    assert(c_r->sibling_left == c_l);
    assert(c_r->sibling_right == NULL);

    assert(c_r->child == NULL);
    assert(c_l->child == c_l_l);

    assert(c_l_l->parent == c_l);
    assert(c_l_r->parent == c_l);

    assert(c_l_l->sibling_left == NULL);
    assert(c_l_l->sibling_right == c_l_r);
    assert(c_l_r->sibling_left == c_l_l);
    assert(c_l_r->sibling_right == NULL);

    free_node(root);
}

void test_free(){
    node_t * root = init_node(1., NULL);
    node_t * c_r = init_node(2., root);
    node_t * c_l = init_node(3., root);

    node_t * c_l_r = init_node(4., c_l);
    node_t * c_l_l = init_node(5., c_l);

    assert(c_l_l->sibling_left == NULL);
    assert(c_l_l->sibling_right == c_l_r);
    free_node(c_l_r);
    assert(c_l_l->sibling_left == NULL);
    assert(c_l_l->sibling_right == NULL);

    assert(c_l->child == c_l_l);
    free_node(c_l_l);
    assert(c_l->child == NULL);

    assert(root->child == c_l);
    free_node(c_l);
    assert(root->child == c_r);
} 

void test_nested_free(){
    node_t * root = init_node(1., NULL);
    node_t * c_r = init_node(2., root);
    node_t * c_l = init_node(3., root);

    init_node(4., c_l);
    init_node(5., c_l);

    assert(root->child == c_l);
    free_node(c_l);
    assert(root->child == c_r);
} 

void test_pass1_case1() {
    node_t * root = init_node(1., NULL);
    init_node(10., root);
    node_t * c_l = init_node(3., root);

    init_node(2., c_l);
    init_node(6., c_l);

    max_sum_pass1(root);
    assert(IS_CLOSE(root->value, 11.));
}

void test_pass1_case2() {
    node_t * root = init_node(4., NULL);
    node_t * c_1 = init_node(4., root);
    node_t * c_2 = init_node(2., root);

    init_node(1., c_1);
    init_node(5., c_1);
    init_node(2., c_1);

    init_node(7., c_2);
    node_t * c_2_2 = init_node(0., c_2);

    init_node(1., c_2_2);
    init_node(4., c_2_2);

    // print_tree(root, 0);
    max_sum_pass1(root);
    assert(IS_CLOSE(root->value, 13.));
    // print_tree(root, 0);
}

void test_pass2_case1(){
    int seq[] = {-1, -1, -1};
    node_t * root = init_node(1., NULL);
    init_node(10., root);
    node_t * c_l = init_node(3., root);

    init_node(2., c_l);
    init_node(6., c_l);

    max_sum_pass1(root);
    max_sum_pass2(root, 0, seq);
    assert(seq[0] == 1);
    assert(seq[1] == 0);
    assert(seq[2] == -1);
}

void test_pass2_case2() {
    int seq[] = {-1, -1, -1, -1};
    node_t * root = init_node(4., NULL);
    node_t * c_1 = init_node(4., root);
    node_t * c_2 = init_node(2., root);

    init_node(1., c_1);
    init_node(5., c_1);
    init_node(2., c_1);

    init_node(7., c_2);
    node_t * c_2_2 = init_node(0., c_2);

    init_node(1., c_2_2);
    init_node(4., c_2_2);

    max_sum_pass1(root);
    max_sum_pass2(root, 0, seq);

    // Case 1:
    int case_1 = (seq[0] == 0) && (seq[1] == 1) && (seq[2] == 0) && (seq[3] == -1);
    int case_2 = (seq[0] == 1) && (seq[1] == 1) && (seq[2] == 0) && (seq[3] == -1);
    assert(case_1 || case_2);
    assert(case_1 != case_2);
}

// void test_children
typedef void (*test_fn_t)();

const test_fn_t fn_list[] = {test_single,
                             test_child, 
                             test_two_children,
                             test_three_children,
                             test_nested_21,
                             test_free,
                             test_nested_free,
                             test_pass1_case1,
                             test_pass1_case2,
                             test_pass2_case1,
                             test_pass2_case2};

const char *fn_names[] = {"test_single", 
                          "test_child",
                          "test_two_children",
                          "test_three_children",
                          "test_nested_21",
                          "test_free",
                          "test_nested_free",
                          "test_pass1_case1",
                          "test_pass1_case2",
                          "test_pass2_case1",
                          "test_pass2_case2"};

int main(int argc, char const *argv[])
{
    test_single();
    for(size_t i = 0; i < 11; ++i){
        printf("[RUN:  %20s]\n", fn_names[i]);
        (fn_list[i])();
        printf("[DONE: %20s]\n", fn_names[i]);
    }
    return 0;
}