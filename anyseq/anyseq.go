// Package anyseq provides APIs for automatic
// differentiation in the context of sequences.
package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Batch represents the output of a sequence batch at a
// single timestep.
//
// The present vectors are packed one after another in a
// single backing vector.
// The absent vectors do not take up any space in the
// backing vector.
type Batch struct {
	// Packed contains the packed vectors.
	Packed anyvec.Vector

	// Present contains one element per sequence in the batch
	// indicating if the corresponding sequence is present.
	// A sequence is called present if it has not terminated
	// before the timestep represented by the Batch.
	//
	// The number of true elements in Present indicates the
	// number of vectors packed in Packed.
	Present []bool
}

// NumPresent counts the true values in Present.
func (b *Batch) NumPresent() int {
	var res int
	for _, x := range b.Present {
		if x {
			res++
		}
	}
	return res
}

// A Seq represents a batch of differentiable sequences.
type Seq interface {
	// Creator returns the underlying vector creator.
	// All sequences must have creators, even if they are
	// empty.
	Creator() anyvec.Creator

	// Output returns the outputs of the batch.
	//
	// It is guaranteed that, if a sequence is not present at
	// timestep t, it will not be present at any timestep
	// after t.
	//
	// No batches should be "empty"--i.e. should have zero
	// timesteps.
	Output() []*Batch

	// Vars returns the variables upon which the output
	// depends.
	Vars() anydiff.VarSet

	// Propagate performs back-propagation through this Seq
	// and all of the non-constant Vecs or Seqs upon which it
	// depends.
	//
	// Propagate may modify the upstream vectors as it likes,
	// but not the present maps.
	Propagate(upstream []*Batch, grad anydiff.Grad)
}
