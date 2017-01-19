// Package anydiff provides a flexible API for automatic
// differentiation.
// It is optimized for machine learning, but it can be
// applied to other areas as well.
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
