package anydifftest

import (
	"fmt"
	"math"
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
	}

	for varIdx, v := range c.Vars() {
		for i := 0; i < v.Vector.Len(); i++ {
			approx := c.Approx(v, i)
			for outIdx, grad := range jacobian {
				actual := anyvec.Sum(grad[v].Slice(i, i+1))
				expected := anyvec.Sum(approx.Slice(outIdx, outIdx+1))
				if !valuesClose(actual, expected, prec) {
					t.Errorf("∂out[%d] / ∂var%d[%d] approximated to %v but got %v",
						outIdx, varIdx, i, expected, actual)
				}
			}
		}
	}
}

func valuesClose(v1, v2 interface{}, prec float64) bool {
	a := valueTo64(v1)
	b := valueTo64(v2)
	if math.IsNaN(a) {
		return math.IsNaN(b)
	} else if math.IsInf(a, 1) {
		return math.IsInf(b, 1)
	} else if math.IsInf(a, -1) {
		return math.IsInf(b, -1)
	} else {
		return math.Abs(a-b) < prec
	}

}

func valueTo64(v interface{}) float64 {
	switch v := v.(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	default:
		panic(fmt.Sprintf("unsupported numeric type: %T", v))
	}
}

func outputCount(c Checker) int {
	for _, v := range c.Vars() {
		if v.Vector.Len() > 0 {
			return c.Approx(v, 0).Len()
		}
	}
	return -1
}
