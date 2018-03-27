#include "node.hpp"

node_t * init_node(const double val, node_t * const parent) 
{
    
    node_t * new_node = (node_t *) malloc(sizeof(node_t));
    if (new_node == (node_t *) NULL)
        return NULL;

    node_t local_node = {.value = val, 
                         .parent = parent,
                         .child = NULL,
                         .sibling_left = NULL,
                         .sibling_right = NULL};
    memcpy(new_node, &local_node, sizeof(local_node));
    if(parent != NULL) {
        if (parent->child != NULL) 
        {
            node_t * old_child = parent->child;
            node_t * old_child_left = old_child->sibling_left;

            new_node->sibling_right = old_child;
            new_node->sibling_left = old_child_left;

            old_child->sibling_left = new_node;
            if (old_child_left != NULL)
                old_child_left->sibling_right = new_node;

        }
        parent->child = new_node;
        parent->num_children += 1;
    }

    return new_node;
}


int free_node(node_t * node) {
    if (node == NULL)
        return 0;

    if (node->parent != NULL) {
        if(node->parent->child == node){
            if (node->sibling_left != NULL)
                node->parent->child = node->sibling_left;
            else if (node->sibling_right != NULL)
                node->parent->child = node->sibling_right;
            else
                node->parent->child = NULL;
        }

        if (node->sibling_right != NULL)
            node->sibling_right->sibling_left = node->sibling_left;

        if (node->sibling_left != NULL)
            node->sibling_left->sibling_right = node->sibling_right;

        node->parent->num_children -= 1;
    }

    for(node_t * child = node->child; child != NULL; child = child->sibling_right)
        free(child);
    free(node);

    return 0;
}

void print_tree(const node_t * const root, const int depth) {
    if (depth == 0)
        printf("%lf\n", root->value);
    else {
        for(int k = 0; k < 3 * (depth - 1); ++k)
            printf(" ");
        printf("|- %lf\n", root->value);
    }

    for (const node_t * child = root->child; child != NULL; child = child->sibling_right)
        print_tree(child, depth + 1);
}