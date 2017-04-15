package anyfwd

import (
	"math"
	"testing"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

const (
	diffDelta   = 1e-4
	diffEpsilon = 1e-3
)

// Tester tests forward automatic-differentiation.
type Tester struct {
	Creator *Creator
	Test    *testing.T
}

// NewTester creates a Tester with an anyvec64 creator.
func NewTester(t *testing.T) *Tester {
	return &Tester{
		Creator: &Creator{
			GradSize:     2,
			ValueCreator: anyvec64.DefaultCreator{},
		},
		Test: t,
	}
}

// TestVecFunc tests that forward auto-diff gives
// approximately correct results for the function f.
func (t *Tester) TestVecFunc(inSize int, f func(in anyvec.Vector) anyvec.Vector) {
	in := t.Creator.ValueCreator.MakeVector(inSize)
	anyvec.Rand(in, anyvec.Normal, nil)
	jacobian := make([]anyvec.Vector, t.Creator.GradSize)
	for i := range jacobian {
		jacobian[i] = t.Creator.ValueCreator.MakeVector(inSize)
		anyvec.Rand(jacobian[i], anyvec.Normal, nil)
	}

	inVec := &Vector{
		CreatorPtr: t.Creator,
		Values:     in,
		Jacobian:   jacobian,
	}
	actualOut := f(inVec.Copy()).Copy().(*Vector)
	expectedOut := t.approxFwdDiff(inVec, f)

	if t.containsNaNs(actualOut) {
		t.Test.Errorf("actual output contains NaNs: %v", actualOut.Data())
	}
	if t.containsNaNs(expectedOut) {
		t.Test.Errorf("expected output contains NaNs: %v", expectedOut.Data())
	}

	if !t.valueVecsClose(actualOut.Values, expectedOut.Values) {
		t.Test.Errorf("value should be %v but got %v", expectedOut.Values.Data(),
			actualOut.Values.Data())
	}

	for i, x := range expectedOut.Jacobian {
		a := actualOut.Jacobian[i]
		if !t.valueVecsClose(x, a) {
			t.Test.Errorf("grad %d should be %v but got %v", i, x.Data(), a.Data())
		}
	}
}

// GetComponent gets a component from a vector.
// The vector may be from a value creator or it may be a
// *Vector.
func (t *Tester) GetComponent(vec anyvec.Vector, idx int) anyvec.Numeric {
	if v, ok := vec.(*Vector); ok {
		res := Numeric{
			Value: t.GetComponent(v.Values, idx),
		}
		for _, grad := range v.Jacobian {
			res.Grad = append(res.Grad, t.GetComponent(grad, idx))
		}
		return res
	} else {
		return anyvec.Sum(vec.Slice(idx, idx+1))
	}
}

// valueVecsClose checks if two vectors from the value
// creator are numerically similar.
func (t *Tester) valueVecsClose(v1, v2 anyvec.Vector) bool {
	diff := v1.Copy()
	diff.Sub(v2)
	switch max := anyvec.AbsMax(diff).(type) {
	case float32:
		return max < diffEpsilon
	case float64:
		return max < diffEpsilon
	default:
		panic("unsupported numeric type")
	}
}

// approxFwdDiff uses finite-differences to perform
// forward auto-diff.
func (t *Tester) approxFwdDiff(in *Vector, f func(in anyvec.Vector) anyvec.Vector) *Vector {
	res := &Vector{CreatorPtr: t.Creator, Values: f(in.Values.Copy()).Copy()}
	for _, grad := range in.Jacobian {
		g := t.approxGrad(in.Values, grad, f)
		res.Jacobian = append(res.Jacobian, g)
	}
	return res
}

func (t *Tester) approxGrad(in, grad anyvec.Vector,
	f func(in anyvec.Vector) anyvec.Vector) anyvec.Vector {
	grad = grad.Copy()
	in = in.Copy()
	grad.Scale(grad.Creator().MakeNumeric(diffDelta))
	in.Add(grad)
	out1 := f(in.Copy()).Copy()
	in.Sub(grad)
	in.Sub(grad)
	out2 := f(in.Copy()).Copy()

	out1.Sub(out2)
	out1.Scale(out1.Creator().MakeNumeric(0.5 / diffDelta))

	return out1
}

func (t *Tester) containsNaNs(vec anyvec.Vector) bool {
	switch data := vec.Data().(type) {
	case []float64:
		for _, x := range data {
			if math.IsNaN(x) {
				return true
			}
		}
		return false
	case NumericList:
		vector := vec.(*Vector)
		if t.containsNaNs(vector.Values) {
			return true
		}
		for _, grad := range vector.Jacobian {
			if t.containsNaNs(grad) {
				return true
			}
		}
		return false
	default:
		panic("unknown vector type")
	}
}
