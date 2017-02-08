package anydifftest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestReverseOut(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq := anyseq.ConstSeqList([][]anyvec.Vector{
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{0, 5})),
			},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{9, 1})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-9, 0})),
			},
		})
		actual := anyseq.Reverse(inSeq)
		expected := anyseq.ConstSeqList([][]anyvec.Vector{
			{
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{0, 5})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{9, 1})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{-9, 0})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
		})
		if len(actual.Output()) != len(expected.Output()) {
			t.Fatalf("expected length %d but got %d", len(expected.Output()),
				len(actual.Output()))
		}
		for i, x := range expected.Output() {
			a := actual.Output()[i]
			if !vectorsClose(getComponents(a.Packed), getComponents(x.Packed), prec) {
				t.Errorf("step %d: expected %v but got %v", i, x.Packed.Data(),
					a.Packed.Data())
			}
			if !reflect.DeepEqual(a.Present, x.Present) {
				t.Errorf("step %d: expected present %v but got %v", i, x.Present,
					a.Present)
			}
		}
	})
}

func TestReverseProp(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Reverse(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
