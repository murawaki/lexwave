#!/bin/env python
# -*- coding: utf-8 -*-
# simple parser of NEXUS annotated trees
import sys
import os
import re

from nexus import NexusReader

class Node(object):
    def __init__(self, _id):
        self._id = _id

class TreeParser(object):
    START = 1
    END = 2
    NODE = 3
    COMMA = 4
    ANNOTATION = 5
    BRANCH = 6
    taxa_re = re.compile(r"([A-Za-z0-9_\-\.]+)")
    branch_re = re.compile(r"(\d+(?:\.\d+))")

    def __init__(self, path):
        self.n = NexusReader()
        self.n.read_file(path)

    def parse(self, _id):
        data = self.n.trees.trees[_id]
        tokens = self._tokenize(data)
        return self._parse(tokens)

    def _tokenize(self, data):
        tree_data = data[data.find('('):-1]  # skip tree-level annotation & strip the last semicolon
        idx = 0
        tokens = []
        while idx < len(tree_data):
            if tree_data[idx] == '(':
                tokens.append(self.START)
                idx += 1
            elif tree_data[idx] == ')':
                tokens.append(self.END)
                idx += 1
            elif tree_data[idx] == ',':
                tokens.append(self.COMMA)
                idx += 1
            elif tree_data[idx] == '[':
                # annotation
                idx2 = tree_data.find(']', idx + 1)
                rawstr = tree_data[idx + 1:idx2]
                annotation = {}
                for kv in rawstr.split(','):
                    k, v = kv.split("=", 1)
                    annotation[k] = v
                obj = {
                    'type': self.ANNOTATION,
                    'annotation': annotation,
                }
                idx = idx2 + 1
                tokens.append(obj)
            elif tree_data[idx] == ':':
                match = self.branch_re.search(tree_data, idx + 1)
                assert(match is not None)
                obj = {
                    'type': self.BRANCH,
                    'branch': float(tree_data[match.start():match.end()]),
                }
                idx = match.end()
                tokens.append(obj)
            else:
                match = self.taxa_re.search(tree_data, idx)
                assert(match is not None)
                taxa = tree_data[match.start():match.end()]
                obj = {
                    'type': self.NODE,
                    'taxa': taxa,
                }
                idx = match.end()
                tokens.append(obj)
        return tokens

    def _parse(self, tokens):
        count = 0
        root = Node(_id=count)
        count += 1
        node = root
        for token in tokens:
            if token == self.START:
                node2 = Node(_id=count)
                count += 1
                node.left = node2
                node2.parent = node
                node = node2
            elif token == self.END:
                node = node.parent
            elif token == self.COMMA:
                node2 = Node(_id=count)
                count += 1
                node.parent.right = node2
                node2.parent = node.parent
                node = node2
            elif token['type'] == self.ANNOTATION:
                node.annotation = token['annotation']
            elif token['type'] == self.BRANCH:
                node.branch = token['branch']
            elif token['type'] == self.NODE:
                node.name = token['taxa']
        return root

if __name__ == "__main__":
    tp = TreeParser(sys.argv[1])
    root = tp.parse(int(sys.argv[2])) # parse the N-th tree
    import cPickle as pickle
    with open(sys.argv[3], "w") as f:
        pickle.dump(root, f)
