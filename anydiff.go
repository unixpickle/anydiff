// Package anydiff provides a flexible automatic
// differentiation package for machine learning and other
// applications.
package anydiff

import "github.com/unixpickle/anyvec"

// A Res is a vector which is capable of performing
// back-propagation through itself and its ancestors.
//
// In general, Vecs should be thought of as immutable and
// thread-safe.
// However, it is not safe to back-propagate through the
// same Grad in multiple Goroutines concurrently.
type Res interface {
	// Output returns the value of the vector.
	Output() anyvec.Vector

	// Vars returns the variables upon which the output
	// depends.
	Vars() VarSet

	// Propagate performs back-propagation through this
	// Result and all of the non-constant results upon which
	// it depends.
	//
	// Propagate may modify the upstream vector as it likes.
	// Often times, the upstream vector can be used as
	// scratch space for computations.
	Propagate(upstream anyvec.Vector, grad Grad)
}

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
	Vars() VarSet

	// Propagate performs back-propagation through this Seq
	// and all of the non-constant Vecs or Seqs upon which it
	// depends.
	//
	// Propagate may modify the upstream vectors as it likes.
	Propagate(upstream []*Batch, grad Grad)
}
