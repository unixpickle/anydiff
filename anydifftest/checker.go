package anydifftest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Checker can compute gradients in two different ways,
// making it possible to perfrom gradient checking.
//
// Behind a Checker, there is some abstract function which
// produces a vector output of a fixed length.
// The gradients with respect to this output vector are
// checked.
//
// The Numerics used by a Checkerd should either be
// float32 or float64.
type Checker interface {
	// Vars returns the variables of interest.
	Vars() []*anydiff.Var

	// Approx approximates the partial derivatives of the
	// output vector with respect to a variable component.
	//
	// This should be done in the least error-prone way,
	// which is typically finite differences.
	Approx(v *anydiff.Var, idx int) anyvec.Vector

	// Exact computes the gradient of a component of the
	// output vector with respect to all the variables.
	Exact(comp int, g anydiff.Grad)
}

// Check performs gradient checking on a Checker.
//
// Differences in the gradients are only considered errors
// if they are greater in magnitude than prec.
func Check(t *testing.T, c Checker, prec float64) {
	n := outputCount(c)
	if n <= 0 {
		return
	}

	jacobian := make([]anydiff.Grad, n)
	for i := range jacobian {
		jacobian[i] = anydiff.NewGrad(c.Vars()...)
		c.Exact(i, jacobian[i])
		if len(jacobian[i]) > len(c.Vars()) {
			t.Error("temporary gradient variables were leaked")
		}
	}

	for varIdx, v := range c.Vars() {
		for i := 0; i < v.Vector.Len(); i++ {
			approx := c.Approx(v, i)
			for outIdx, grad := range jacobian {
				actual := getComponent(grad[v], i)
				expected := getComponent(approx, outIdx)
				if !valuesClose(actual, expected, prec) {
					t.Errorf("∂out[%d] / ∂var%d[%d] approximated to %v but got %v",
						outIdx, varIdx, i, expected, actual)
				}
			}
		}
	}
}

// CheckVars runs one sub-test per variable.
// In each subtest, only one variable is checked.
// This is useful for testing that back-propagation is not
// over-aggressively avoiding constants.
func CheckVars(t *testing.T, c Checker, prec float64) {
	for i, v := range c.Vars() {
		t.Run(fmt.Sprintf("Vars[%d]", i), func(t *testing.T) {
			c := &checkerSubset{c, v}
			Check(t, c, prec)
		})
	}
}

type checkerSubset struct {
	Checker
	V *anydiff.Var
}

func (c *checkerSubset) Vars() []*anydiff.Var {
	return []*anydiff.Var{c.V}
}

func outputCount(c Checker) int {
	for _, v := range c.Vars() {
		if v.Vector.Len() > 0 {
			return c.Approx(v, 0).Len()
		}
	}
	return -1
}
