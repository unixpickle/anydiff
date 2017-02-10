package anydifftest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// A SeqChecker is a Checker for any function that returns
// an anyseq.Seq.
//
// It requires that the numeric type is either float32 or
// float64.
type SeqChecker struct {
	F func() anyseq.Seq
	V []*anydiff.Var

	// Delta is the finite difference to use when computing
	// approximate partials.
	// If it is 0, a default is used.
	Delta float64

	// Prec is the error precision.
	// If an error is less than Prec in magnitude, it is
	// ignored.
	// If Prec is 0, then a default for the numeric type is
	// used.
	Prec float64
}

// FullCheck runs several variations of gradient checking.
func (v *SeqChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		Check(t, v, v.prec())
	})
	CheckVars(t, v, v.prec())
	t.Run("Accumulated", func(t *testing.T) {
		v1 := *v
		v1.F = func() anyseq.Seq {
			return accumulateSeq(v.F())
		}
		Check(t, &v1, v1.prec())
	})
}

// Vars returns v.V.
func (v *SeqChecker) Vars() []*anydiff.Var {
	return v.V
}

// Approx approximates the partial derivatives of the
// output vector using finite differences.
func (v *SeqChecker) Approx(variable *anydiff.Var, idx int) anyvec.Vector {
	delta := v.delta()
	old := getComponent(variable.Vector, idx)
	setComponent(variable.Vector, idx, old+delta)
	posOut := packSeqOut(v.F().Output())
	setComponent(variable.Vector, idx, old-delta)
	negOut := packSeqOut(v.F().Output())
	setComponent(variable.Vector, idx, old)

	posOut.Sub(negOut)
	posOut.Scale(posOut.Creator().MakeNumeric(1 / (2 * delta)))
	return posOut
}

// Exact computes the exact gradient of an output
// component using automatic differentiation.
func (v *SeqChecker) Exact(comp int, g anydiff.Grad) {
	out := v.F()
	if out.Creator() == nil {
		panic("missing creator")
	}
	if !g.Intersects(out.Vars()) {
		return
	}
	upstream := oneHotBatches(out.Output(), comp)
	out.Propagate(upstream, g)
}

func (v *SeqChecker) delta() float64 {
	if v.Delta != 0 {
		return v.Delta
	}
	return v.prec()
}

func (v *SeqChecker) prec() float64 {
	if v.Prec != 0 {
		return v.Prec
	}
	return v.defaultPrec()
}

func (v *SeqChecker) defaultPrec() float64 {
	var n anyvec.Numeric
	if len(v.V) > 0 {
		n = v.V[0].Vector.Creator().MakeNumeric(0)
	} else {
		n = v.F().Output()[0].Packed.Creator().MakeNumeric(0)
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

func packSeqOut(b []*anyseq.Batch) anyvec.Vector {
	var concatMe []anyvec.Vector
	for _, x := range b {
		concatMe = append(concatMe, x.Packed)
	}
	return concatMe[0].Creator().Concat(concatMe...)
}

type accumulatorSeq struct {
	Out []*anyseq.Batch
	In  anyseq.Seq
}

func accumulateSeq(in anyseq.Seq) anyseq.Seq {
	out := copyBatches(in.Output())
	scaleBatches(out, 4)
	return &accumulatorSeq{
		Out: out,
		In:  in,
	}
}

func (a *accumulatorSeq) Creator() anyvec.Creator {
	return a.In.Creator()
}

func (a *accumulatorSeq) Output() []*anyseq.Batch {
	return a.Out
}

func (a *accumulatorSeq) Vars() anydiff.VarSet {
	return a.In.Vars()
}

func (a *accumulatorSeq) Propagate(u []*anyseq.Batch, g anydiff.Grad) {
	scaleBatches(u, 2)
	a.In.Propagate(copyBatches(u), g)
	a.In.Propagate(u, g)
}

func copyBatches(b []*anyseq.Batch) []*anyseq.Batch {
	r := make([]*anyseq.Batch, len(b))
	for i, x := range b {
		r[i] = &anyseq.Batch{Packed: x.Packed.Copy(), Present: x.Present}
	}
	return r
}

func scaleBatches(b []*anyseq.Batch, s float64) {
	for _, x := range b {
		x.Packed.Scale(x.Packed.Creator().MakeNumeric(s))
	}
}

func oneHotBatches(b []*anyseq.Batch, idx int) []*anyseq.Batch {
	res := make([]*anyseq.Batch, len(b))
	for i, x := range b {
		if idx >= 0 && idx < x.Packed.Len() {
			oh := make([]float64, x.Packed.Len())
			oh[idx] = 1
			numList := x.Packed.Creator().MakeNumericList(oh)
			p := x.Packed.Creator().MakeVectorData(numList)
			res[i] = &anyseq.Batch{Packed: p, Present: x.Present}
		} else {
			zero := x.Packed.Creator().MakeVector(x.Packed.Len())
			res[i] = &anyseq.Batch{Packed: zero, Present: x.Present}
		}
		idx -= x.Packed.Len()
	}
	return res
}
