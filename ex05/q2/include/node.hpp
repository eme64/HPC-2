#ifndef NODE_HPP
#define NODE_HPP

#include <cstdlib>
#include <cstring>
#include <cstdio>

/*
    Node structure used for the Markov tree.

    An invariant that should be honored is:
    node.parent == NULL implies that node.sibling_left and node.sibling_right
    are undefined. This ensures a unique root node.

    Attributes
    ----------
        value : double
            The numerical value assigned to the node.

        parent : node_t *
            The pointer to the parent node. NULL if node is a root.

        child : node_t *
            Pointer to one of the children. Subsequent children are
            accessed via their attributes sibling_left and _right.
        
        sibling_left, sibling_right : node_t *
            Pointers to the siblings, constituting a linked list.

 */
struct node_t {
    double value;
    node_t * const parent;
    node_t *child, *sibling_left, *sibling_right;
    int num_children;
};

/*
    Create the node with a given value and parent.

    The routine allocates the necessary memory for the node
    and set its attributes. It also modifies the parents attributes
    to reference the newly created node.

    If parent == NULL, the new_node is considered the root of a tree.
    Otherwise, parent is modified as follows. parent->child will be set 
    to new_node, and  parent->child->sibling_left, 
    parent->child->sibling_right updated in accordance with doubly 
    linked list rules.

    When a non-NULL pointer is returned, following attributes invariants
    hold:
    new_node->child == NULL
    new_node->value == val
    new_node->parent == parent

    Parameters
    ----------
        val : double
            The value to be stored at new_node.value
        parent : node_t *
            The pointer to the parent, to be stored at new_node->parent.

    Returns
    -------
        new_node : node_t *
            The pointer to the newly created node. 
            NULL is returned when memory cannot be allocated for the new 
            node.
 */
node_t * init_node(const double val, node_t * const parent);

/*
    Deallocate the node and all its children.

    The routine recursively deallocates the memory assigned to the
    node and its children. 

    It changes the node->sibling_left->sibling_right, 
    node->sibling_right->sibling_left attributes according to the 
    doubly linked list element removal rules. 

    If the parent->child == node, parent->child will be set to one of 
    the remaining siblings.

    Parameters
    ----------
        node : node_t *
            The node to be deallocated.

    Returns
    -------
        err_code : int
            Error code, 0 on success.
 */
int free_node(node_t * node);

/*
    Print a human-friendly representation of the tree.

    Parameters
    ----------
        node : node_t *
            The node whose value and children are to be printed
        offset : int
            The offset from left side of the terminal. Used internally
            due to recursion. Pass 0.
 */
void print_tree(const node_t * const root, const int offset);

#endif // NODE_HPP

