#pragma once

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>

#include <vector>

namespace faiss {

struct CodePacker;

/** Index that adds all vectors without copying by storing pointers to each
 *  batch of data. Based off of IndexFlatCodes */
struct IndexRefCodes : Index {
    size_t code_size;

    /// encoded datasets
    std::vector<uint8_t *> code_storage;

    /// the last ids in each storage entry
    std::vector<idx_t> end_ids;

    IndexRefCodes();

    IndexRefCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /* Search which storage batch the key is in using binary search
    * and return the index of the key in that batch */
    void get_indices(idx_t key, idx_t * storage_batch, idx_t * storage_index) const;

    /* Merge batch and the next batch to reduce binary search cost.
    * Requires that batch is not the last batch. Implemented to
    * only work with adjacent batches to keep the ordering of ids. */
    void merge_batch(idx_t batch);

    size_t sa_code_size() const override;

    /** remove some ids. NB that because of the structure of the
     * index, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;

    /** a FlatCodesDistanceComputer offers a distance_to_code method */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

    // returns a new instance of a CodePacker
    CodePacker* get_CodePacker() const;

    void check_compatible_for_merge(const Index& otherIndex) const override;

    virtual void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    // permute_entries. perm of size ntotal maps new to old positions
    void permute_entries(const idx_t* perm);
};

} // namespace faiss