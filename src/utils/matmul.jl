function matmul!(output,a,b)
    output .= a*b
end

function elmul!(output,a,b)
    output .= a.*b
end