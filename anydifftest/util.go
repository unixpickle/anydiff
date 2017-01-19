package anydifftest

import (
	"fmt"
	"math"

	"github.com/unixpickle/anyvec"
)

func valuesClose(a, b, prec float64) bool {
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

func getComponent(v anyvec.Vector, i int) float64 {
	switch data := v.Slice(i, i+1).Data().(type) {
	case []float32:
		return float64(data[i])
	case []float64:
		return data[i]
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
