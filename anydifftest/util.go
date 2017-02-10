package anydifftest

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

// SeqsClose returns true if the two sequences are
// numerically equivalent up to the given precision.
func SeqsClose(a, b anyseq.Seq, prec float64) bool {
	if len(a.Output()) != len(b.Output()) {
		return false
	}
	for i, x := range a.Output() {
		y := b.Output()[i]
		if !vectorsClose(getComponents(x.Packed), getComponents(y.Packed), prec) {
			return false
		}
		if !reflect.DeepEqual(x.Present, y.Present) {
			return false
		}
	}
	return true
}

func runWithCreators(t *testing.T, f func(t *testing.T, c anyvec.Creator, prec float64)) {
	t.Run("float32", func(t *testing.T) {
		f(t, anyvec32.DefaultCreator{}, defaultPrec32)
	})
}

func valuesClose(a, b, prec float64) bool {
	if math.IsNaN(a) {
		return math.IsNaN(b)
	} else if math.IsInf(a, 1) {
		return math.IsInf(b, 1)
	} else if math.IsInf(a, -1) {
		return math.IsInf(b, -1)
	} else {
		mag := math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
		return math.Abs(a-b) < prec*mag
	}
}

func vectorsClose(a, b []float64, prec float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, aVal := range a {
		if !valuesClose(aVal, b[i], prec) {
			return false
		}
	}
	return true
}

func getComponents(v anyvec.Vector) []float64 {
	switch data := v.Data().(type) {
	case []float64:
		return data
	case []float32:
		res := make([]float64, len(data))
		for i, x := range data {
			res[i] = float64(x)
		}
		return res
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
}

func getComponent(v anyvec.Vector, i int) float64 {
	switch data := v.Slice(i, i+1).Data().(type) {
	case []float32:
		return float64(data[0])
	case []float64:
		return data[0]
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
}

func setComponent(v anyvec.Vector, i int, val float64) {
	data := v.Data()
	switch data := data.(type) {
	case []float32:
		data[i] = float32(val)
	case []float64:
		data[i] = val
	default:
		panic(fmt.Sprintf("unsupported type: %T", data))
	}
	v.SetData(data)
}

func makeRandomVec(c anyvec.Creator, size int) *anydiff.Var {
	v := c.MakeVector(size)
	anyvec.Rand(v, anyvec.Normal, nil)
	return anydiff.NewVar(v)
}

func makeAbsFriendlyVec(c anyvec.Creator, size int) *anydiff.Var {
	v := anydiff.NewVar(c.MakeVector(size))
	anyvec.Rand(v.Vector, anyvec.Uniform, nil)
	v.Vector.AddScaler(c.MakeNumeric(0.1))
	mask := c.MakeVector(size)
	anyvec.Rand(mask, anyvec.Bernoulli, nil)
	mask.AddScaler(c.MakeNumeric(-0.5))
	v.Vector.Mul(mask)
	return v
}

func makeDivisionFriendlyVec(c anyvec.Creator, size int) *anydiff.Var {
	v := c.MakeVector(size)
	anyvec.Rand(v, anyvec.Uniform, nil)
	v.AddScaler(c.MakeNumeric(0.25))
	return anydiff.NewVar(v)
}

func makeBasicTestSeqs(c anyvec.Creator) (anyseq.Seq, []*anydiff.Var) {
	batches := []*anyseq.ResBatch{
		{
			Packed:  makeRandomVec(c, 24),
			Present: []bool{true, true, true, true},
		},
		{
			Packed:  makeRandomVec(c, 18),
			Present: []bool{true, true, true, false},
		},
		{
			Packed:  makeRandomVec(c, 12),
			Present: []bool{true, false, true, false},
		},
		{
			Packed:  makeRandomVec(c, 12),
			Present: []bool{true, false, true, false},
		},
		{
			Packed:  makeRandomVec(c, 6),
			Present: []bool{false, false, true, false},
		},
		{
			Packed:  makeRandomVec(c, 6),
			Present: []bool{false, false, true, false},
		},
	}
	var varList []*anydiff.Var
	for _, x := range batches {
		for v := range x.Packed.Vars() {
			varList = append(varList, v)
		}
	}
	return anyseq.ResSeq(c, batches), varList
}
