// Package anyseq provides APIs for automatic
// differentiation in the context of sequences.
package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Batch is a set of equally-sized vectors which are
// packed one-after-another in a larger vector.
type Batch struct {
	// Vector contains the packed vectors.
	Vector anyvec.Vector

	// Num specifies the number of vectors.
	// It must evenly divide Vector.Len().
	Num int
}

// A Seq represents a batch of differentiable sequences.
// It is a batched, sequence analog for Vec.
type Seq interface {
	// Output returns the outputs of the sequence at each
	// timestep.
	Output() []*Batch

	// Vars returns the variables upon which the output
	// depends.
	Vars() anydiff.VarSet

	// Propagate performs back-propagation through this Seq
	// and all of the non-constant Vecs or Seqs upon which it
	// depends.
	//
	// Propagate may modify the upstream vectors as it likes.
	Propagate(upstream []*Batch, grad anydiff.Grad)
}
