package anydifftest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

const (
	defaultPrec32 = 1e-3
	defaultPrec64 = 1e-5
)

// A VecChecker is a Checker for any function that returns
// an anydiff.Vec.
//
// It requires that the numeric type is either float32 or
// float64.
type VecChecker struct {
	F func() anydiff.Vec
	V []*anydiff.Var

	// Delta is the finite difference to use when computing
	// approximate partials.
	// If it is 0, the error precision is used as the delta.
	Delta float64

	// Prec is the error precision.
	// If an error is less than Prec in magnitude, it is
	// ignored.
	// If Prec is 0, then a default for the numeric type is
	// used.
	Prec float64

	numericSample anyvec.Numeric
}

// FullCheck runs several variations of gradient checking.
func (v *VecChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		Check(t, v, v.prec())
	})
	CheckVars(t, v, v.prec())
	t.Run("Accumulated", func(t *testing.T) {
		v1 := *v
		v1.F = func() anydiff.Vec {
			return accumulate(v.F())
		}
		Check(t, &v1, v1.prec())
	})
}

// Vars returns v.V.
func (v *VecChecker) Vars() []*anydiff.Var {
	return v.V
}

// Approx approximates the partial derivatives of the
// output vector using finite differences.
func (v *VecChecker) Approx(variable *anydiff.Var, idx int) anyvec.Vector {
	delta := v.delta()
	old := getComponent(variable.Vector, idx)
	setComponent(variable.Vector, idx, old+delta)
	posOut := v.F().Output()
	setComponent(variable.Vector, idx, old-delta)
	negOut := v.F().Output()
	setComponent(variable.Vector, idx, old)

	posOut.Sub(negOut)
	posOut.Scale(posOut.Creator().MakeNumeric(1 / (2 * delta)))
	return posOut
}

// Exact computes the exact gradient of an output
// component using automatic differentiation.
func (v *VecChecker) Exact(comp int, g anydiff.Grad) {
	out := v.F()
	oneHot := make([]float64, out.Output().Len())
	oneHot[comp] = 1
	nums := out.Output().Creator().MakeNumericList(oneHot)
	upstream := out.Output().Creator().MakeVectorData(nums)
	out.Propagate(upstream, g)
}

func (v *VecChecker) delta() float64 {
	if v.Delta != 0 {
		return v.Delta
	}
	return v.prec()
}

func (v *VecChecker) prec() float64 {
	if v.Prec != 0 {
		return v.Prec
	}
	return v.defaultPrec()
}

func (v *VecChecker) defaultPrec() float64 {
	var n anyvec.Numeric
	if len(v.V) > 0 {
		n = v.V[0].Vector.Creator().MakeNumeric(0)
	} else {
		n = v.F().Output().Creator().MakeNumeric(0)
	}
	switch n := n.(type) {
	case float32:
		return defaultPrec32
	case float64:
		return defaultPrec64
	default:
		panic(fmt.Sprintf("unsupported numeric type: %T", n))
	}
}

type accumulatorRes struct {
	OutVec anyvec.Vector
	In     anydiff.Vec
}

func accumulate(in anydiff.Vec) anydiff.Vec {
	v := in.Output().Copy()
	v.Scale(v.Creator().MakeNumeric(4))
	return &accumulatorRes{
		OutVec: v,
		In:     in,
	}
}

func (a *accumulatorRes) Output() anyvec.Vector {
	return a.OutVec
}

func (a *accumulatorRes) Vars() anydiff.VarSet {
	return a.In.Vars()
}

func (a *accumulatorRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	u.Scale(u.Creator().MakeNumeric(2))
	a.In.Propagate(u.Copy(), g)
	a.In.Propagate(u, g)
}
