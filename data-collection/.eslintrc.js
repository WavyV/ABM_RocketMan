module.exports = {
    extends: [
        'airbnb',
        'plugin:import/errors',
        'plugin:import/warnings',
    ],
    globals: {
        document: true,
        window: true,
    },
    parser: 'babel-eslint',
    rules: {
        'no-nested-ternary': 'warn',
        'no-unused-vars': 'warn',
        'sort-imports': 'error',
        'react/forbid-prop-types': 'warn',
        'react/prefer-stateless-function': 'warn',
    },
    settings: {
        'import/resolver': 'webpack',
    },
};
