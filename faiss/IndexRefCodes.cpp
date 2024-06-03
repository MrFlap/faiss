/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRefCodes.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

IndexRefCodes::IndexRefCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexRefCodes::IndexRefCodes() : code_size(0) {}

void IndexRefCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    code_storage.push_back((uint8_t *)x);
    ntotal += n;
    end_ids.push_back(ntotal);
    //codes.resize((ntotal + n) * code_size);
    //sa_encode(n, x, codes.data() + (ntotal * code_size));
    //ntotal += n;
}

void IndexRefCodes::reset() {
    // NOTE: this clears the memory that stores the vectors.
    for(int i = 0; i < code_storage.size(); i++) {
        if(code_storage[i]) {
            free(code_storage[i]);
        }
    }
    code_storage.clear();
    end_ids.clear();
    ntotal = 0;
}

void IndexRefCodes::get_indices(idx_t key, idx_t * storage_batch, idx_t * storage_index) const {
    size_t lo = 0;
    size_t hi = code_storage.size() - 1;
    while(lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if(end_ids[mid] > key) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    *storage_batch = lo;
    if(lo == 0) {
        *storage_index = key;
    } else {
        *storage_index = key - end_ids[lo - 1];
    }
}

size_t IndexRefCodes::sa_code_size() const {
    return code_size;
}

size_t IndexRefCodes::remove_ids(const IDSelector& sel) {
    size_t total = 0;
    size_t prev_id = 0;
    for (idx_t code_seg = 0; code_seg < code_storage.size(); code_seg++){
        uint8_t * codes = code_storage[code_seg];
        idx_t total = end_ids[code_seg] - prev_id;
        idx_t j = 0;
        for (idx_t i = 0; i < total; i++) {
            if (!sel.is_member(i)) {
                if (i > j) {
                    memmove(&codes[code_size * j],
                            &codes[code_size * i],
                            code_size);
                }
                j++;
            }
        }
        ntotal += j;
        prev_id = end_ids[code_seg];
        end_ids[code_seg] = prev_id + j;
    }
    size_t nremove = ntotal - total;
    ntotal = total;
    return nremove;
}

void IndexRefCodes::merge_batch(idx_t batch) {
    if(batch == code_storage.size() - 1) {
        return;
    }
    idx_t batch1 = batch;
    idx_t batch2 = batch + 1;
    idx_t size1 = end_ids[batch1];
    if(batch1 > 0) {
        size1 -= end_ids[batch1 - 1];
    }
    idx_t size2 = end_ids[batch2] - end_ids[batch1];
    uint8_t * new_alloc = (uint8_t *)malloc(code_size * (size1 + size2));
    FAISS_THROW_IF_NOT(new_alloc);
    memcpy(new_alloc, code_storage[batch1], code_size * size1);
    memcpy(new_alloc + code_size * size1, code_storage[batch2], code_size * size2);
    free(code_storage[batch1]);
    free(code_storage[batch2]);
    code_storage[batch1] = new_alloc;
    end_ids[batch1] = end_ids[batch2];
    code_storage.erase(code_storage.begin() + batch2);
}


void IndexRefCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    //sa_decode(ni, codes.data() + i0 * code_size, recons);
    idx_t start_batch;
    idx_t start_batch_index;
    idx_t end_batch;
    idx_t end_batch_index;
    get_indices(i0, &start_batch, &start_batch_index);
    get_indices(i0 + ni - 1, &end_batch, &end_batch_index);
    idx_t total_copied = 0;
    for(idx_t i = start_batch; i <= end_batch; i++) {
        idx_t start_index = 0;
        idx_t end_index = end_ids[i] - 1;
        if(i > 0) {
            end_index -= end_ids[i - 1];
        }
        if(i == start_batch) {
            start_index = start_batch_index;
        }
        if(i == end_batch) {
            end_index = end_batch_index;
        }
        idx_t to_copy = end_index - start_index + 1;
        uint8_t * codes = code_storage[i];
        sa_decode(to_copy, codes + start_index * code_size, recons + total_copied * code_size);
        total_copied += to_copy;
    }
}

void IndexRefCodes::reconstruct(idx_t key, float* recons) const {
    //reconstruct_n(key, 1, recons);
    idx_t batch;
    idx_t index;
    get_indices(key, &batch, &index);
    uint8_t * codes = code_storage[batch];
    sa_decode(1, codes + index * code_size, recons);
}

FlatCodesDistanceComputer* IndexRefCodes::get_FlatCodesDistanceComputer()
        const {
    FAISS_THROW_MSG("not implemented");
}

void IndexRefCodes::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexRefCodes* other =
            dynamic_cast<const IndexRefCodes*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
}

void IndexRefCodes::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in RefCodes index");
    check_compatible_for_merge(otherIndex);
    IndexRefCodes* other = static_cast<IndexRefCodes*>(&otherIndex);
    for(idx_t i = 0; i < other->code_storage.size(); i++) {
        code_storage.push_back(other->code_storage[i]);
    }
    ntotal += other->ntotal;
    other->reset();
}

CodePacker* IndexRefCodes::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

void IndexRefCodes::permute_entries(const idx_t* perm) {
    uint8_t * new_codes = (uint8_t *)malloc(code_size * ntotal);
    FAISS_THROW_IF_NOT(new_codes);

    idx_t storage_batch = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        while(end_ids[storage_batch] <= i) {
            storage_batch++;
        }
        idx_t storage_id = i;
        if(storage_batch > 0) {
            storage_id -= end_ids[storage_batch - 1];
        }
        memcpy(new_codes + i * code_size,
               code_storage[storage_batch] + perm[i] * code_size,
               code_size);
    }
    for(idx_t i = 0; i < code_storage.size(); i++) {
        free(code_storage[i]);
    }
    code_storage.clear();
    end_ids.clear();
    code_storage.push_back(new_codes);
    end_ids.push_back(ntotal);
}

} // namespace faiss
